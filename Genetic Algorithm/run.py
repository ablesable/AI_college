# Bartosz Soból 215881
# Algorytm Genetyczny

from random import randint
import random
import sys
import math


def adaptation_function(adaptation_values, people):
    for person in people:
        adaptation_values.append(2 * (person * person + 1))


def random_start_value(people, number):
    for x in range(number):
        people.append(randint(1, 127))


def line_chances(chances, adaptation_values, adaptation_sum):
    for value in adaptation_values:
        chances.append(value / adaptation_sum)


def quadratic_chances(chances, adaptation_values, adaptation_sum):
    for value in adaptation_values:
        chances.append((value * value) / (adaptation_sum * adaptation_sum))


def root_chances(chances, adaptation_values, adaptation_sum):
    for value in adaptation_values:
        chances.append(math.sqrt(value) / math.sqrt(adaptation_sum))


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1


def mutation(survivors, mutation_probability):
    mutation_count = 0
    for idx, person in enumerate(
        survivors
    ):  # for nonprimitive types. For accessing list which we iterate
        random_value = random.uniform(0, 1)

        if random_value > mutation_probability:
            continue

        elif random_value < mutation_probability:  # MUTTATION OCCURES
            lottery = random.randint(2, 7)  # choose between 2 and 7th bit
            list_bit_person = list(
                bin(person)
            )  # making a list from a string for iteration. Forming '0bxxxxxxx'

            print("Mutation Occured, Value before mutation: " + str(person))

            if list_bit_person[lottery] == "0":
                list_bit_person[lottery] = "1"
            else:
                list_bit_person[lottery] = "0"

            list_bit_person.remove("b")  # removing b char from list of characters
            bit_string = listToString(list_bit_person)  # making it string again
            survivors[idx] = int(bit_string, 2)
            mutation_count += 1
            print("Value after mutation")
            print(str(survivors[idx]))

    print("Mutation occured " + str(mutation_count) + " times")


def crossing(survivors, crossing_probability):
    children = []
    for i in range(0, len(survivors), 2):
        pair = []

        try:
            pair.append(survivors[i])
            pair.append(survivors[i + 1])
            random_value = random.uniform(0, 1)  # will or will not be crossing

            if random_value > crossing_probability:  # will not be crossing
                children.append(survivors[i])
                children.append(survivors[i + 1])
                continue

            else:
                parent_one = list(bin(pair[0]))
                parent_one.remove("b")
                parent_two = list(bin(pair[1]))
                parent_two.remove("b")

                genetic_len_one = len(parent_one)
                genetic_len_two = len(parent_two)

                lottery = random.randint(
                    1, genetic_len_one - 1
                )  # crossing genomes lottery (between second and penultimate element)

                child_one = parent_one[:lottery] + parent_two[lottery:]
                child_two = parent_two[:lottery] + parent_one[lottery:]

                child_one_string = listToString(child_one)
                child_two_string = listToString(child_two)

                children.append(int(child_one_string, 2))
                children.append(int(child_two_string, 2))

        except:
            print("No pair for this element.")
            break
    print("After crossing, this is next Generation: ")
    print(children)
    print("______________________________")
    print("\n")
    return children


# MAIN
print("Please enter how many generation you want?")
generation_count = int(input())

print("Please choose roulette selection. Type LINE, QUADRA or ROOT")
roulette = input()
print("You choose " + roulette + " mode")

print("Please choose probability of mutation. Type a value between 0.1 to 0.9")
mutation_probability = float(input())
print("You choose " + str(mutation_probability) + " chances of mutation")

print("Please choose probability of crossing. Type a value between 0.1 to 0.9")
crossing_probability = float(input())
print("You choose " + str(crossing_probability) + " chances of crossing")

people = []  # random, start values between 1 and 127 here

POPULATION = 14

for x in range(generation_count):
    print("GENERATION " + str(x + 1))
    print("This is entry population in this generation")
    print(people)
    if x == 0:
        random_start_value(people, POPULATION)

    adaptation_values = []  # adaptation values here
    adaptation_function(adaptation_values, people)

    adaptation_sum = sum(adaptation_values)  # sum whole adaptation

    # chances of survive:
    # ROULETTE LINE
    if roulette == "LINE":
        chances = []
        line_chances(chances, adaptation_values, adaptation_sum)
        roulette_list = random.choices(people, chances, k=POPULATION // 2)
        duplicate_roulette_list = roulette_list.copy()
        survivors = roulette_list + duplicate_roulette_list
        print("Survivors chosen in line-roulette:")
        print(survivors)

        # ROULETTE QUADRATIC
    elif roulette == "QUADRA":
        chances = []
        quadratic_chances(chances, adaptation_values, adaptation_sum)
        roulette_list = random.choices(people, chances, k=POPULATION // 2)
        duplicate_roulette_list = roulette_list.copy()
        survivors = roulette_list + duplicate_roulette_list
        print("Survivors chosen in quadratic-roulette:")
        print(survivors)

    # ROULETTE ROOT
    elif roulette == "ROOT":
        chances = []
        root_chances(chances, adaptation_values, adaptation_sum)
        roulette_list = random.choices(people, chances, k=POPULATION // 2)
        duplicate_roulette_list = roulette_list.copy()
        survivors = roulette_list + duplicate_roulette_list
        print("Survivors chosen in root-roulette:")
        print(survivors)

    else:
        print("Wrong Argument!")
        sys.exit()

    # mutation:
    mutation(survivors, mutation_probability)
    print("Survivors after mutation")
    print(survivors)

    print("\n")
    # crossing
    children = crossing(survivors, crossing_probability)
    people = children