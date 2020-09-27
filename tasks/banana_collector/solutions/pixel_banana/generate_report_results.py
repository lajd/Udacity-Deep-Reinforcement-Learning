import os
from tools.generate_report import save_report


if __name__ == '__main__':
    save_report(
        project_title="Visual Banana Results",
        report_save_dir='RESULTS.pdf',
        tunings_directory=os.path.abspath('visual_tunings'),
        performance_threshold=10.0
    )
