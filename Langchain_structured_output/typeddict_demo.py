from typing import TypedDict

class Person(TypedDict):

    name: str
    age : int

# create a new dict on this format
# new_person is a type of Person dict
new_person : Person = {'name': "monika", 'age': 35}

# here we have one issue:
# if we will change 35 to '35'(str)still this code will run:
# that means we can define the format but TypedDict() cann't validate it.
# like the entered values are  entered with correct datatype or not
print(new_person)