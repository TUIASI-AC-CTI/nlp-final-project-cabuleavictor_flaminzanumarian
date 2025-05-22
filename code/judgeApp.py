import sys
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QMessageBox, QHBoxLayout, QGroupBox
)

from hate_detector_and_llm import HateSpeechDetector, LLMJudge

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LLMJudgeThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, llm_judge, result):
        super().__init__()
        self.llm_judge = llm_judge
        self.result = result

    def run(self):
        try:
            explanation = self.llm_judge.judge(self.result)
            self.finished_signal.emit(explanation)
        except Exception as e:
            self.error_signal.emit(str(e))


class HateSpeechApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hate Speech Detector")
        self.setGeometry(100, 100, 1800, 1400)
        self.detector1 = HateSpeechDetector("GroNLP/hateBERT", "model_epoch_20.pt")
        self.llm_judge = LLMJudge()
        self.detector2 = HateSpeechDetector("GroNLP/hateBERT")
        self._init_ui()
        self.llm_thread = None

    def _init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1c1c1e;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 22px;  /* Increased from 18px */
            }
            QTextEdit, QLabel {
                background-color: #2c2c2e;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 16px;  /* Increased padding */
                font-size: 22px;  /* Increased from 18px */
            }
            QTextEdit:read-only {
                background-color: #3a3a3c;
            }
            QPushButton {
                background-color: #ff9500;
                color: black;
                font-size: 25px;  /* Increased from 20px */
                font-weight: bold;
                padding: 16px 28px;  /* Increased padding */
                border: none;
                border-radius: 8px;
                min-width: 200px;  /* Increased min-width */
            }
            QPushButton:hover {
                background-color: #ffad33;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 10px;
                margin-top: 20px;  /* Increased margin */
                padding-top: 30px;  /* Increased padding */
                background-color: #2c2c2e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 12px;  /* Increased padding */
                font-weight: bold;
                font-size: 25px;  /* Increased from 20px */
                color: #ff9500;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        input_group = QGroupBox("Input Text")
        input_layout = QVBoxLayout()
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setStyleSheet("font-size: 26px;")
        input_layout.addWidget(self.text_input)
        input_group.setLayout(input_layout)
        input_group.setMaximumHeight(180)
        layout.addWidget(input_group)

        button_layout = QHBoxLayout()
        self.classify_btn1 = QPushButton("Analyze with HateBERT Fine-Tuned")
        self.classify_btn1.clicked.connect(self.classify_text1)

        self.llm_judge_btn = QPushButton("Explain with LLM - Mistral")
        self.llm_judge_btn.clicked.connect(self.llm_judgment)
        self.llm_judge_btn.setEnabled(False)

        self.classify_btn2 = QPushButton("Analyze with original HateBERT")
        self.classify_btn2.clicked.connect(self.classify_text2)

        button_layout.addWidget(self.classify_btn1)
        button_layout.addWidget(self.llm_judge_btn)
        button_layout.addWidget(self.classify_btn2)
        layout.addLayout(button_layout)

        results_layout = QHBoxLayout()

        result_group1 = QGroupBox("HateBERT Fine-Tuned Results")
        result_layout1 = QVBoxLayout()
        self.result_label1 = QLabel("Awaiting input...")
        self.result_label1.setWordWrap(True)
        self.result_label1.setStyleSheet("font-size: 22px;")
        self.figure1 = Figure(facecolor="#2c2c2e")
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas1.setMaximumHeight(300)
        result_layout1.addWidget(self.result_label1)
        result_layout1.addWidget(self.canvas1)
        result_group1.setLayout(result_layout1)

        result_group2 = QGroupBox("Original HateBERT Results")
        result_layout2 = QVBoxLayout()
        self.result_label2 = QLabel("Awaiting input...")
        self.result_label2.setWordWrap(True)
        self.result_label2.setStyleSheet("font-size: 22px;")
        self.figure2 = Figure(facecolor="#2c2c2e")
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setMaximumHeight(300)
        result_layout2.addWidget(self.result_label2)
        result_layout2.addWidget(self.canvas2)
        result_group2.setLayout(result_layout2)

        results_layout.addWidget(result_group1)
        results_layout.addWidget(result_group2)
        layout.addLayout(results_layout)

        llm_group = QGroupBox("LLM Explanation")
        llm_layout = QVBoxLayout()
        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)
        self.llm_output.setMaximumHeight(400)
        self.llm_output.setStyleSheet("font-size: 26px;")
        llm_layout.addWidget(self.llm_output)
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)

        self.setLayout(layout)

    def classify_text1(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text.")
            return
        try:
            result = self.detector1.make_prediction(text)
            self.latest_result = result
            self.result_label1.setText(
                f"<b>Prediction:</b> {result['prediction']}<br>"
                f"<b>Confidence:</b> {result['confidence']:.2%}<br>"
                f"<b>Hate Probability:</b> {result['hate_probability']:.4f}<br>"
                f"<b>Not Hate Probability:</b> {result['not_hate_probability']:.4f}"
            )
            self.plot_probabilities(self.figure1, self.canvas1, result['hate_probability'],
                                    result['not_hate_probability'], "HateBERT Fine-Tuned")
            self.llm_judge_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
            self.llm_judge_btn.setEnabled(False)

    def classify_text2(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text.")
            return
        try:
            result = self.detector2.make_prediction(text)
            self.result_label2.setText(
                f"<b>Prediction:</b> {result['prediction']}<br>"
                f"<b>Confidence:</b> {result['confidence']:.2%}<br>"
                f"<b>Hate Probability:</b> {result['hate_probability']:.4f}<br>"
                f"<b>Not Hate Probability:</b> {result['not_hate_probability']:.4f}"
            )
            self.plot_probabilities(self.figure2, self.canvas2, result['hate_probability'],
                                    result['not_hate_probability'], "Original HateBERT")
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def plot_probabilities(self, figure, canvas, hate_prob, not_hate_prob, title_suffix):
        figure.clear()
        ax = figure.add_subplot(111)
        labels = ['Hate', 'Not Hate']
        values = [hate_prob, not_hate_prob]
        bars = ax.bar(labels, values, color=['#ff3b30', '#34c759'])

        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(f'Prediction Probabilities - {title_suffix}', color='white', fontsize=16)
        ax.set_facecolor('#2c2c2e')
        ax.tick_params(colors='white', labelsize=14)
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                    f'{height:.2f}', ha='center', color='white', fontweight='bold', fontsize=14)

        canvas.draw()

    def llm_judgment(self):
        if not hasattr(self, 'latest_result'):
            QMessageBox.warning(self, "Error", "Please classify text first.")
            return

        self.llm_judge_btn.setEnabled(False)
        self.llm_output.setPlainText("Processing explanation from LLM...")

        self.llm_thread = LLMJudgeThread(self.llm_judge, self.latest_result)
        self.llm_thread.finished_signal.connect(self.llm_judgment_finished)
        self.llm_thread.error_signal.connect(self.llm_judgment_error)
        self.llm_thread.finished.connect(self.llm_thread_cleanup)
        self.llm_thread.start()

    def llm_judgment_finished(self, explanation):
        self.llm_output.setPlainText(explanation)
        self.llm_judge_btn.setEnabled(True)

    def llm_judgment_error(self, error_msg):
        QMessageBox.critical(self, "LLM Error", error_msg)
        self.llm_judge_btn.setEnabled(True)

    def llm_thread_cleanup(self):
        self.llm_thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HateSpeechApp()
    window.show()
    sys.exit(app.exec_())
