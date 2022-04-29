import ghostscript
from utils.download.parsertools.wordcorrector import WordCorrector
from multiprocessing import Process


class PDFParser:
    def __init__(self, dictionary_path: str, registry) -> None:
        self.dictionary_path = dictionary_path
        self.registry = registry

    def parse(self, file):

        temp_file = "utils/temp.txt"

        p = Process(target=self.read, args=(file, temp_file,))
        print("Starting Ghostscript...")
        p.start()
        p.join(30)
        if p.is_alive():
            p.terminate()
            print("Ghostscript timed out.")
            return None, {"error": True}

        with open(temp_file) as file:
            full_text = file.read()

        corrector = WordCorrector(dictionary_path=self.dictionary_path)
        new_text, corrector_results = corrector.correct_text(full_text)

        return new_text, corrector_results

    def read(self, file, temp_file):

        args = [
            "-q",
            "-sDEVICE=txtwrite",
            "-dNOPAUSE",
            "-dSAFER",
            f'-sOutputFile="{temp_file}"',
            f'"{file}"',
        ]

        args = [a.encode("UTF-8") for a in args]

        with ghostscript.Ghostscript(*args) as g:
            ghostscript.cleanup()


"""
        reader = PdfFileReader(file)

        full_text = ""
        for page_num in range(reader.numPages):
            full_text += reader.pages[page_num].extractText() + "\n\n"
            
            """
