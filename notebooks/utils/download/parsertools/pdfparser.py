from multiprocessing import Process
from pathlib import Path
import ghostscript
from utils.download.parsertools.wordcorrector import WordCorrector


class PDFParser:
    def __init__(self, registry) -> None:
        self.registry = registry

    def parse(self, file):

        temp_folder = Path(__file__).parent.resolve() / "temp"
        temp_folder.mkdir(exist_ok=True)

        p = Process(
            target=self.read,
            args=(
                file,
                temp_folder,
            ),
        )
        print("Starting Ghostscript...")
        p.start()
        p.join(30)
        if p.is_alive():
            p.terminate()
            print("Ghostscript timed out.")
            return None, {"error": True}

        full_text = ""

        for path in temp_folder.iterdir():
            with open(path) as file:
                full_text += " " + file.read()

            # We delete the files to avoid having extra pages in the folder for the next file.
            path.unlink()

        corrector = WordCorrector()
        new_text, corrector_results = corrector.correct_text(full_text)

        return new_text, corrector_results

    def read(self, file, temp_folder):
        """
        Extracts text from a PDF file using Ghostscript.

        We define the parameters in a list and pass them to the interface.

        Importantly, we save each page on a separate file using the Ghostscript specification.
        We will later read each page separately. This avoids some strange behavior from
        Ghostscript where it only saved the last page of some PDF files.
        """

        args = [
            "-q",
            "-dNEWPDF",
            "-sDEVICE=txtwrite",
            "-dNOPAUSE",
            "-dSAFER",
            f'-sOutputFile="{temp_folder}/temp_pg_%d.txt"',
            f'"{file}"',
        ]

        args = [a.encode("UTF-8") for a in args]

        with ghostscript.Ghostscript(*args) as g:
            ghostscript.cleanup()


if __name__ == "__main__":
    parser = PDFParser(None)
    parser.parse("dummy.pdf")
