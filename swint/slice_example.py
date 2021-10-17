mystr = 'TutorialsTeacher'
nums = [1,2,3,4,5,6,7,8,9,10]

portion1 = slice(9)
portion2 = slice(0, -2, None)

print('slice: ', portion1)
print('String value: ', mystr[portion1])
print('List value: ', nums[portion1])

print('slice: ', portion2)
print('String value: ', mystr[portion2])
print('List value: ', nums[portion2])


nums = [1,2,3,4,5,6,7,8,9,10]
odd_portion = slice(0, 10, 2)
print(nums[odd_portion])

even_portion = slice(1, 10, 2)
print(nums[even_portion])