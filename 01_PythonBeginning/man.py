class Man:
    def __init__(self, name):
        self.name = name
        print("init!")

    def hello(self):
        print("hello, " + self.name)

m = Man("David")
m.hello()
