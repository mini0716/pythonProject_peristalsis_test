python_score = [57, 86, 63, 92, 35, 79]
a = 1
for i in python_score :
    if i >= 80 :
        grade = "A"
    elif i >= 60 :
        grade = "B"
    else  :
        grade = "C"

    print(f"{a}번은 {i}점 이며, {grade}등급 입니다.")
    a += 1