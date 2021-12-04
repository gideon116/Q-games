class Players:

    def __init__(self, size, restriction=None):

        self.size = size
        self.restriction = restriction

        if self.restriction is not None:
            retry = True

            while retry:
                potential_x = random.choice(range(self.size))
                potential_y = random.choice(range(self.size))

                for ip in self.restriction:

                    # if rand number lands on restriction
                    if potential_x == ip[0] and potential_y == ip[1]:
                        retry = True
                        break

                    else:
                        retry = False

                self.x = potential_x
                self.y = potential_y

        else:
            self.x = random.choice(range(self.size))
            self.y = random.choice(range(self.size))

    def pos(self):

        position = (self.x, self.y)
        return position

    def movement(self, choice):

        if choice == 0:
            self.new_pos(x=1, y=1)
        elif choice == 1:
            self.new_pos(x=1, y=-1)

        elif choice == 2:
            self.new_pos(x=-1, y=-1)
        elif choice == 3:
            self.new_pos(x=-1, y=1)

        elif choice == 4:
            self.new_pos(x=0, y=0)
        elif choice == 5:
            self.new_pos(x=1, y=0)

        elif choice == 6:
            self.new_pos(x=0, y=1)
        elif choice == 7:
            self.new_pos(x=-1, y=0)

        elif choice == 8:
            self.new_pos(x=0, y=-1)

    def new_pos(self, x=None, y=None):

        if self.restriction is not None:

            # if x (and so y) is not specified, choose random numbers avoiding the restriction
            if x is None:
                retry = True
                potential_x = None
                potential_y = None

                while retry:
                    potential_x = self.x + random.choice([-1, 0, 1])  # either -1 or 1
                    potential_y = self.x + random.choice([-1, 0, 1])

                    # if it tries to go out of bounds
                    if potential_x < 0:
                        potential_x = 0
                    elif potential_x > self.size - 1:
                        potential_x = self.size - 1

                    # if it tries to go out of bounds
                    if potential_y < 0:
                        potential_y = 0
                    elif potential_y > self.size - 1:
                        potential_y = self.size - 1

                    for ix in self.restriction:

                        # if it tries to pass some restriction
                        if potential_x == ix[0] and potential_y == ix[1]:
                            retry = True
                            break

                        else:
                            retry = False

                self.x = potential_x
                self.y = potential_y

            else:
                on_restriction = False

                potential_x = self.x + x
                potential_y = self.y + y

                # if it tries to go out of bounds
                if potential_x < 0:
                    potential_x = 0
                elif potential_x > self.size - 1:
                    potential_x = self.size - 1

                # if it tries to go out of bounds
                if potential_y < 0:
                    potential_y = 0
                elif potential_y > self.size - 1:
                    potential_y = self.size - 1

                for ixx in self.restriction:

                    # if it tries to pass some restriction
                    if potential_x == ixx[0] and potential_y == ixx[1]:
                        on_restriction = True
                        break

                    else:
                        on_restriction = False

                if on_restriction:
                    self.x += 0
                    self.y += 0
                else:
                    self.x = potential_x
                    self.y = potential_y

        else:
            if x is None:
                self.x += random.choice([-1, 0, 1])
            else:
                self.x += x

            if y is None:
                self.y += random.choice([-1, 0, 1])  # either -1 or 1
            else:
                self.y += y

            # if it tries to go out of bounds
            if self.x < 0:
                self.x = 0
            elif self.x > self.size - 1:
                self.x = self.size - 1

            # if it tries to go out of bounds
            if self.y < 0:
                self.y = 0
            elif self.y > self.size - 1:
                self.y = self.size - 1
