import re
import os
import sys
from PyPDF2 import PdfFileReader


class DelBrokenPDF:
    def __init__(self, path):
        self.filter = [".pdf"]
        self.root_path = path

    def isValidPDF_pathfile2(self, pathfile):
        bValid = True
        try:
            # PdfFileReader(open(pathfile, 'rb'))
            reader = PdfFileReader(pathfile)
            if reader.getNumPages() < 1:
                bValid = False
        except:
            bValid = False
            # print('*' + traceback.format_exc())

        return bValid

    def isValidPDF_pathfile(self, pathfile):
        content = ''
        with open(pathfile, mode='rb') as f:
            content = f.read()
        partBegin = content[0:20]
        if partBegin.find(rb'%PDF-1.') < 0:
            print('Error: not find %PDF-1.')
            return False

        idx = content.rfind(rb'%%EOF')
        if idx < 0:
            print('Error: not find %%EOF')
            return False

        partEnd = content[(0 if idx - 100 < 0 else idx - 100): idx + 5]
        if not re.search(rb'startxref\s+\d+\s+%%EOF$', partEnd):
            print('Error: not find startxref')
            return False

        return True

    def all_path(self, dirname):

        result = []

        for maindir, subdir, file_name_list in os.walk(dirname):

            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]

                if ext in self.filter:
                    result.append(apath)

        return result

    def start(self):
        # root_path = "../abstractive_papers"

        print(self.root_path)
        dir_or_files = self.all_path(self.root_path)
        with open(os.path.join(self.root_path, "result.txt"), "w") as fp:
            for dir_file in dir_or_files:
                if self.isValidPDF_pathfile(dir_file):
                    if not self.isValidPDF_pathfile2(dir_file):
                        print("ERROR2: ", dir_file)
                        fp.write("ERROR\t" + dir_file + "\r\n")
                        os.remove(dir_file)
                else:
                    print("ERROR: ", dir_file)
                    fp.write("ERROR\t" + dir_file + "\r\n")
                    os.remove(dir_file)


OUTPUT_DIR = '../check_data/'
del_pdf = DelBrokenPDF(OUTPUT_DIR)
del_pdf.start()
