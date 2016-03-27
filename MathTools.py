class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector( self.x + other.x, self.y + other.y, self.z + other.z )

    def __mul__(self, scale):
        return scale*self

    def __rmul__(self, scale):
        return Vector( scale*self.x, scale*self.y, scale*self.z )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector(  self.y*other.z - self.z*other.y,
                        self.z*other.x - self.x*other.z,
                        self.x*other.y - self.y*other.x )

    def assign(self, other):
        self.x = other.x
        self.y = other.y
        self.z = other.z
