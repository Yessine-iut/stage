# f = open("list_of_created_pronouns_tobecleaned", "r")

# all_pronouns = []
# for pronoun in f.readlines():
# 	all_pronouns.append(pronoun)
# # print(len(all_pronouns))
# # print(len(set(all_pronouns)))
# clean_list = set(all_pronouns)

# out = open("list_of_pronouns.txt","a") 
# for pronoun in clean_list:
# 	print(pronoun)
# 	out.write(pronoun)

f = open("list_of_pronouns.txt", "r")
print(len(f.readlines()))