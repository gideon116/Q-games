class Wall:

    def __init__(self, env_size, area=None):

        self.area_covered = area
        self.env_size = env_size
        self.pos_range = None

    def location(self, location):

        if location == 'special':

            self.pos_range = [(5, 0), (5, 1), (5, 3), (5, 4), (5, 7), (5, 8), (5, 9), (0, 0), (0, 1), (0, 3), (0, 4),
                              (0, 5), (0, 6), (0, 7), (0, 8), (7, 6), (8, 6), (9, 6)]

        elif location == 'left':

            # NOT USED FOR NOW
            self.pos_range = [(0, self.env_size - 1), (1, self.env_size - 1), (2, self.env_size - 1),
                              (3, self.env_size - 1)]

        elif location == 'top':

            # NO 4
            self.pos_range = [(3, 2), (3, 3), (3, 4), (3, 5)]

        elif location == 'bottom':

            # NO 2, 3, 4, 5
            self.pos_range = [(7, 0), (7, 1), (7, 6), (7, 7), (7, 8), (7, 9)]

        elif location == 'middle':

            # NO 6
            self.pos_range = [(self.env_size // 2, 0), (self.env_size // 2, 1), (self.env_size // 2, 2),
                              (self.env_size // 2, 3), (self.env_size // 2, 4), (self.env_size // 2, 5),
                              (self.env_size // 2, 7), (self.env_size // 2, 8), (self.env_size // 2, 9)]
