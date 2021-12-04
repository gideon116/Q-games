class Lava:

    #  THE Q-TABLE IS DESIGNED FOR ONLY 4 LAVA BLOCKS. IF YOU WANT TO CHANGE THE SIZE OF THE LAVA, REDESIGN THE Q-TABLE
    def __init__(self, env_size):
        self.env_size = env_size
        self.pos_range = None

    def number_of_lava(self, location):
        if location == 'one' or location == 'One' or location == 'ONE':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3)]

        elif location == 'two' or location == 'Two' or location == 'TWO':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3), (6, 2), (6, 3), (7, 2), (7, 3)]

        elif location == 'three' or location == 'Three' or location == 'THREE':
            self.pos_range = [(1, 2), (1, 3), (2, 2), (2, 3), (6, 2), (6, 3), (7, 2), (7, 3),
                              (1, 8), (1, 9), (2, 8), (2, 9)]
