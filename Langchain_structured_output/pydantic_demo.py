from pydantic import BaseModel,EmailStr,Field
from typing import Optional

# optional used: if no values is passed for any key in the defined dict then it sholud take the default value
# pydantic gets inherited from BaseModel class

#EmailStr : another function available in pydantic used to validated the correct email address
#Field : used to put some constrain(check/limit/filter)

class Student(BaseModel):
    name: str
    age : Optional[int] = None   # if user will not pass any value to age it will be None and this "age" is optional value here
    email : EmailStr
    cgpa: float = Field(gt=0,lt =10,default = 5)  # gt-->greater than , lt--->less than
new_student = {'name':"Monika",'age':32,'email':'abc@gmail.com'}
# it will not alow any other datatype value to pass as name like 32 or somthing else which was possible in Typeddict
# it will through an error that means it do the data validation as well.

student = Student(**new_student)

print(student)
print(student.name)

# can convert it into dict:
student_dict = dict(student)
print(student_dict["name"])

# can make it json:
student_json = student.model_dump_json()