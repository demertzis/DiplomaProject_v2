class Charge():
    def __init__(self):
        self.charge_list = [1,2]


class Garage():
    def __init__(self, charge_list: Charge = []):
        self.charge_list = charge_list
    def print_list(self):
        print(self.charge_list.charge_list)

    def empty_list(self):
        self.charge_list.charge_list = []

var1 = Charge()
var2 = var3 = Garage(var1)
var2.print_list()
var3.print_list()
var2.empty_list()

var2.print_list()
var3.print_list()

var2.charge_list.charge_list = [12,2,3]

var2.print_list()
var3.print_list()
