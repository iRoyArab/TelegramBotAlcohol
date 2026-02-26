import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path

class IndicatorExtractor:
    def __init__(self, logs_dir):
        self.logs_dir = Path(logs_dir)
        self.scan_logs = defaultdict(list)
        self.suspicious_scans = set()
        self.total_logs_processed = 0
        self.total_lines_analyzed = 0
        
        # Regex Patterns
        self.rx_scan_id = re.compile(r'(?:scan ID |scanId = |\[scan-id:)(\w+)', re.IGNORECASE)
        self.rx_score = re.compile(r'Score:\s*([0-9.]+)', re.IGNORECASE)
        self.rx_threat_name = re.compile(r'(?:threat name = |threatName: )[\'"]?([^\'"\s,]+)', re.IGNORECASE)
        self.rx_behavioral = re.compile(r'threat = ([^\s,]+)', re.IGNORECASE)
        self.rx_file_path = re.compile(r'(?:path = |scanning:\s*)["\']?([^"\'\,]+)', re.IGNORECASE)
        self.rx_verdict = re.compile(r'final verdict:\s*(\w+)', re.IGNORECASE)
        # זיהוי חתימה דיגיטלית עבור משימה 2
        self.rx_signature = re.compile(r'Signature verified:\s*([^,]+),\s*signed\s*=\s*(\w+)', re.IGNORECASE)

    def process_all_logs(self):
        if not self.logs_dir.exists():
            print(f"Error: Directory {self.logs_dir} not found.")
            return

        for filepath in self.logs_dir.glob('*.log'):
            self.total_logs_processed += 1
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        self._process_line(line.strip(), filepath.name)
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='utf-16') as file:
                    for line in file:
                        self._process_line(line.strip(), filepath.name)

    def _process_line(self, line, filename):
        self.total_lines_analyzed += 1
        match_id = self.rx_scan_id.search(line)
        if not match_id:
            return
        
        scan_id = match_id.group(1)
        self.scan_logs[scan_id].append({"file": filename, "content": line})

        score_match = self.rx_score.search(line)
        if score_match:
            try:
                # ניקוי הנקודה בסוף המספר אם קיימת
                clean_score = score_match.group(1).rstrip('.')
                if float(clean_score) > 0.75:
                    self.suspicious_scans.add(scan_id)
            except ValueError:
                pass

        if self.rx_threat_name.search(line) or self.rx_behavioral.search(line):
            self.suspicious_scans.add(scan_id)

    def generate_report(self, output_path):
        report = {
            "suspiciousIndicators": [],
            "summary": {
                "totalLogsProcessed": self.total_logs_processed,
                "totalScansAnalyzed": len(self.scan_logs),
                "suspiciousIndicatorsFound": len(self.suspicious_scans)
            }
        }

        for scan_id in self.suspicious_scans:
            lines = self.scan_logs[scan_id]
            
            indicator = {
                "scanId": scan_id,
                "classification": "true_positive", # ברירת מחדל עד שנוכיח אחרת
                "detectionType": "Unknown",
                "suspiciousFile": None,
                "threatName": None,
                "mlScore": None,
                "sourceLogFile": lines[0]["file"],
                "extraDetails": {}
            }

            for log_entry in lines:
                line = log_entry["content"]
                
                fp_match = self.rx_file_path.search(line)
                if fp_match: indicator["suspiciousFile"] = fp_match.group(1).strip()
                
                tn_match = self.rx_threat_name.search(line)
                if tn_match: 
                    indicator["threatName"] = tn_match.group(1).rstrip('.')
                    indicator["detectionType"] = "Static File Scanner"
                
                bh_match = self.rx_behavioral.search(line)
                if bh_match:
                    indicator["threatName"] = bh_match.group(1).rstrip('.')
                    indicator["detectionType"] = "Behavioral Engine"

                sc_match = self.rx_score.search(line)
                if sc_match: 
                    indicator["mlScore"] = sc_match.group(1).rstrip('.')
                    indicator["detectionType"] = "Machine Learning"
                
                vd_match = self.rx_verdict.search(line)
                if vd_match: indicator["extraDetails"]["finalVerdict"] = vd_match.group(1)

                # שליפת חתימה דיגיטלית
                sig_match = self.rx_signature.search(line)
                if sig_match:
                    indicator["extraDetails"]["signatureVendor"] = sig_match.group(1).strip()
                    indicator["extraDetails"]["signatureValid"] = sig_match.group(2).strip().upper()

            # --- לוגיקת סיווג False Positive (משימה 2) ---
            file_path_lower = (indicator["suspiciousFile"] or "").lower()
            threat_lower = (indicator["threatName"] or "").lower()
            vendor = indicator["extraDetails"].get("signatureVendor", "").lower()
            is_valid_sig = indicator["extraDetails"].get("signatureValid") == "VALID"
            verdict_lower = (indicator["extraDetails"].get("finalVerdict") or "").lower()

            # סדר התנאים קריטי: מהתנאי המנצח והחזק ביותר, ועד לחלש ביותר
            if is_valid_sig and ("microsoft" in vendor or "google" in vendor or "validvendor" in vendor) and ("system32" in file_path_lower or "program files" in file_path_lower):
                # 1. חתום, חברה מוכרת, במיקום לגיטימי של המערכת = 100% שווא
                indicator["classification"] = "false_positive"
                
            elif "pua" in threat_lower or "bundler" in threat_lower or "pua" in verdict_lower:
                # 2. סרגלי כלים/תוכנות זבל = דורש בירור
                indicator["classification"] = "uncertain"
                
            elif is_valid_sig and ("microsoft" in vendor or "google" in vendor or "validvendor" in vendor):
                # 3. חתום, אבל זרוק בתיקיית הורדות או מיקום מוזר = דורש בירור
                indicator["classification"] = "uncertain"
                
            else:
                # 4. אם זה לא חתום, לא PUA, והציון חריג/יש שם של וירוס = איום אמיתי
                indicator["classification"] = "true_positive"

            report["suspiciousIndicators"].append(indicator)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(report, outfile, indent=2)
        
        print(f"Report generated successfully: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract suspicious indicators from security logs.')
    parser.add_argument('--logs-dir', type=str, default=r"C:\home_assignment\logs", help='Directory containing log files')
    parser.add_argument('--output', type=str, default='report.json', help='Output JSON file path')
    args = parser.parse_args()
    
    extractor = IndicatorExtractor(args.logs_dir)
    extractor.process_all_logs()
    extractor.generate_report(args.output)

if __name__ == "__main__":
    main()
