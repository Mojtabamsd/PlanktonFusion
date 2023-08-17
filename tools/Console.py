import sys
import datetime
import os
import traceback


class Console:
    def __init__(self, log_file="console_log.txt", error_file="error_log.txt"):
        self.log_file = log_file
        self.error_file = error_file

        # Redirect unhandled exceptions to custom error handler
        sys.excepthook = self.handle_exception

    def log(self, *args, log_type="log"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = " ".join(str(arg) for arg in args)
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        if log_type == "log":
            with open(self.log_file, "a") as f:
                f.write(formatted_message + "\n")
        elif log_type == "error":
            with open(self.error_file, "a") as f:
                f.write(formatted_message + "\n")

    def warning(self, *args):
        self.log("WARNING:", *args, log_type="warning")

    def error(self, *args):
        self.log("ERROR:", *args, log_type="error")

    def info(self, *args):
        self.log("INFO:", *args)

    def quit(self, exit_code=0):
        sys.exit(exit_code)

    def save_log(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, self.log_file)
        with open(self.log_file, "r") as f:
            log_contents = f.read()
            with open(output_file, "w") as output_f:
                output_f.write(log_contents)
        print(f"Log saved to {output_file}")

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if exc_type is not KeyboardInterrupt:  # Don't log keyboard interrupt
            self.error("An unhandled exception occurred:")
            self.log("\n".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)), log_type="error")