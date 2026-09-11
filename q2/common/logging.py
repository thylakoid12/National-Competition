class Tee:
    def __init__(self,stream,file):
        self.stream,self.file=stream,file
    def write(self,text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()

