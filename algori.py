from operator import indexOf

print("1번-------------------------------------")

numbers = [4,8,15,16,23,42]

total = sum(numbers)
avg = total / len(numbers)

print(total)
print(avg)

print("2번-------------------------------------")

nums = [3,5,2,9,1]
total = sum(nums)
maxone = max(nums)
minone = min(nums)

print(total)
print(maxone)
print(minone)

print("3번-------------------------------------")

arr = [1,2,3,4,5,6,7,8,9,10]
odd = len(arr[::2])
even = len(arr[1::2])

print(odd)
print(even)

print("4번-------------------------------------")

numList = list(range(2,11,2))
ListSum = sum(numList)
ListAvg = ListSum / len(numList)

print(ListSum)
print(ListAvg)

print("5번-------------------------------------")

nList = [5,9,1,7,3]
nMax = max(nList)
nMin = min(nList)

print(nMax)
print(nMin)

print("6번-------------------------------------")

import random

nCard = [1,2,3,4,5,6,7,8,9,10]
random.shuffle(nCard)
print(nCard)

oneCard_index = nCard.index(1)
twoCard_index = nCard.index(2)

nCard.remove(oneCard_index)
nCard.remove(twoCard_index)

sumCard = oneCard_index + twoCard_index

print(nCard)
print(oneCard_index)
print(twoCard_index)
print(f"sum = {sumCard}")

if sumCard > 10:
    print(f"{sumCard}이(가) 10보다 크므로 승리!")
else:
    print(f"{sumCard}이(가) 10보다 작거나 같으므로 실패 ㅠㅠ")