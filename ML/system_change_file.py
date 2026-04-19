import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QMessageBox
)

ENV_FILE = os.path.join(os.path.dirname(__file__), "../.env")


def read_env():
    env_data = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_data[key] = value.strip('"')
    return env_data


def write_env(env_data):
    with open(ENV_FILE, "w") as f:
        for key, value in env_data.items():
            f.write(f'{key}="{value}"\n')


class EnvEditor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ENV Editor")
        self.setGeometry(100, 100, 300, 150)

        self.env_data = read_env()

        layout = QVBoxLayout()

        # Label
        self.label = QLabel("Email:")
        layout.addWidget(self.label)

        # Input field
        self.input = QLineEdit()
        self.input.setText(self.env_data.get("email", ""))
        layout.addWidget(self.input)

        # Save button
        self.button = QPushButton("Save")
        self.button.clicked.connect(self.save_env)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def save_env(self):
        email = self.input.text().strip()

        if email:
            self.env_data["email"] = email
        else:
            self.env_data["email"] = self.env_data.get("email", "")

        write_env(self.env_data)

        QMessageBox.information(self, "Success", ".env updated!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnvEditor()
    window.show()
    sys.exit(app.exec_())