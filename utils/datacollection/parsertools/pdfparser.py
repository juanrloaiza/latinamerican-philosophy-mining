import ghostscript
from datacollection.parsertools.wordcorrector import WordCorrector


class PDFParser:
    def __init__(self, dictionary_path: str, registry) -> None:
        self.dictionary_path = dictionary_path
        self.registry = registry

    def parse(self, file):

        temp_file = "temp.txt"
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

        with open(temp_file) as file:
            full_text = file.read()

        corrector = WordCorrector(dictionary_path=self.dictionary_path)
        new_text, corrector_results = corrector.correct_text(full_text)

        return new_text, corrector_results


"""
        reader = PdfFileReader(file)

        full_text = ""
        for page_num in range(reader.numPages):
            full_text += reader.pages[page_num].extractText() + "\n\n"
            
            """
