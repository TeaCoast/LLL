class Hello:
    def __getitem__(self, index):
        print(index)


hello = Hello()
hello[0]
hello[0, 1, 2]
hello[(0, 1, 2), [2, 3, 4]]