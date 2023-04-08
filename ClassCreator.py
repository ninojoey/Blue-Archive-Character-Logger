className = ""
classParameters = []
defaultParameters = []

def printClassAndParameters():
    print("Class name: ")
    print("1. {}".format(className))
    print("Class parameters: ")
    for i in range(len(classParameters)):
        print("{}. {}:{}".format((i + 2), classParameters[i], defaultParameters[i]))


    

INDENT_STRING = "    "

className = input("Enter your class name: ")
parametersInput = input("Enter all parameters (paramName0,defaultVal0 paramName1,defaultVal1 ...): ")

parametersInput = parametersInput.split(" ")
for i in range(len(parametersInput)):
    parameterInput = parametersInput[i].split(",")

    if len(parameterInput) == 2:
        classParameters.append(parameterInput[0])
        defaultParameters.append(parameterInput[1])
    else:
        classParameters.append(parameterInput[0])
        defaultParameters.append(0)
        

printClassAndParameters()







changeParamInput = input("Enter the option you'd like to change (0 to pass): ")
while changeParamInput != "0":
    if changeParamInput == "1":
        className = input("Enter your new class name: ")
        
        printClassAndParameters()
        changeParamInput = input("Would you like to make any further changes? (0 for no): ")
    else:
        paramIndex = int(changeParamInput) - 2
        print("parameter to change: ", classParameters[paramIndex])
        
        editParamInput = input("1. Edit \n2. Delete\nEnter which option you'd like to do: ")
        if editParamInput == "1":
            editParamInput = input("1. Change Name\n2. Change defualt value\n3. Change Order\nEnter which option you'd like to do: ")
            if editParamInput == "1":
                newParamName = input("Enter a new parameter name: ")
                classParameters[paramIndex] = newParamName
            elif editParamInput == "2":
                newDefaultValue = input("Enter the new default value: ")
                defaultParameters[paramIndex] = newDefaultValue
            else:
                newParamIndex = input("Enter a new parameter index (starting with 1): ")
                newParamIndex = int(newParamIndex)
                newParamIndex -= 1
                parameterValue = classParameters.pop(paramIndex)

                if newParamIndex > len(classParameters):
                    classParameters.append(parameterValue)
                else:
                    classParameters.insert(newParamIndex, parameterValue)

        else:
            print("Removed value: ", classParameters.pop(paramIndex))

        printClassAndParameters()
        changeParamInput = input("Would you like to make any further changes? (0 for no): ")
    
classHeader = "class {}:\n".format(className)
defaultConstructor = "{}def __init__(self):\n".format(INDENT_STRING)
for i in range(len(classParameters)):
    defaultConstructor += "{}{}self.{} = {}\n".format(INDENT_STRING, INDENT_STRING, classParameters[i], defaultParameters[i])

paramConstructorHead = "\n{}def __init__(self".format(INDENT_STRING, className)
paramConstructorBody = ""
for i in range(len(classParameters)):
    classParameter = classParameters[i]
    paramConstructorHead += ", {}".format(classParameter)
    paramConstructorBody += "{}{}self.{} = {}\n".format(INDENT_STRING, INDENT_STRING, classParameter, classParameter)

paramConstructorHead += "):\n"
paramConstructor = paramConstructorHead + paramConstructorBody

classToText = classHeader + defaultConstructor + paramConstructor

print(classToText)
    
