def prompt_eg(row):
    """
    nballs:
        6 or 10
        total # of balls in each cage. 
    ndraws_from_cage:
        6 or 7
        # of draws from the cage, i.e., in the outcome.
    cage_A_balls_marked_N:
        4
        # of N balls in cage A
    cage_B_balls_marked_N:
        3 or 6
        # of N balls in cage B
    nballs_prior_cage:
        6 or 10
        # of die sides
    priors:
        2, 3, 4, 6, 7
        a draw of 1 to priors will result in cage A, otherwise cage B
    ndraws:
        0 - 7
        # of N balls in the outcome
    """

    # Introduction
    if row['pay'] == 1:
        introduction = "You are participating in a decision-making experiment, where you can earn money based on the number of correct decisions you make."
    elif row['pay'] == 0:
        introduction = "You are participating in a decision-making experiment."
    else:
        raise ValueError('Invalid pay value.')
        
    # Explanation
    explanation = \
f"""

There are two identical bingo cages, Cage A and Cage B, each containing {row['nballs']} balls. Cage A contains {row['cage_A_balls_marked_N']} balls labeled "N" and {row['nballs'] - row['cage_A_balls_marked_N']} balls labeled "G", while Cage B contains {row['cage_B_balls_marked_N']} balls labeled "N" and {row['nballs'] - row['cage_B_balls_marked_N']} balls labeled "G".

A {row['nballs_prior_cage']}-sided die is used to determine which of the two cages will be used to generate draws. If a random roll of the die shows 1 through {row['priors']}, I will use Cage A; if it shows {row['priors'] + 1} through {row['nballs_prior_cage']}, I will use Cage B. You will not know the outcome of the roll of the die or which cage I use.

Once a cage is chosen at random based on the roll of the die, it is used to generate draws with replacement.

I have drawn a total of {row['ndraws_from_cage']} balls with replacement. The result is {row['ndraws']} "N" balls and {row['ndraws_from_cage'] - row['ndraws']} "G" balls.
After observing this outcome, which cage do you think generated the observations? Your decision is correct if the balls were drawn from that cage.
"""

    # Instructions
    if row['instruction'] == 'reasoning':
        instruction = \
"""
YOU ARE WELCOME TO ALSO DESCRIBE YOUR REASONING, BROKEN INTO SEPARATE STEPS, TO EXPLAIN HOW YOU ARRIVED AT YOUR FINAL ANSWER. 
Please state your answer in the following format at the end.
"Final answer: Cage A." or "Final answer: Cage B.".
"""
    elif row['instruction'] == 'no reasoning':
        instruction = \
"""
PLEASE JUST REPORT YOU FINAL ANSWER AND DO NOT PROVIDE ANY REASONING AS TO HOW YOU ARRIVED AT YOUR FINAL ANSWER. 
Please state your answer in the following format.
"Final answer: Cage A." or "Final answer: Cage B.".
""" 
    else:
        raise ValueError('Invalid instruction value.')

    return introduction + explanation + instruction


def prompt_hs(row):

    """
    Prior Pr(A):
        1/2 or 2/3
        prior probability of cage A
    ndraws_from_cage:
        1, 2, 3, 4, 7
        # of draws from the cage, i.e., in the outcome.
    """

    # Introduction
    introduction = "This is an experiment in the economics of decision making. Various agencies have provided funds for the experiment. Your earnings will depend partly on your decisions and partly on chance. If you are careful and make good decisions, you may earn a considerable amount of money, which will be paid to you, privately, in cash, at the end of the experiment. In addition to the money that you earn during the experiment, you will also receive $6. This payment is to compensate you for showing up today."

    # Explanation
    if row['Prior Pr(A)'] == "1/2":
        prior_A_threshold = 3
    elif row['Prior Pr(A)'] == "2/3":
        prior_A_threshold = 4
    else:
        raise ValueError(f'Invalid prior value {row['prior']}')
    
    if row['ndraws_from_cage'] == 1:
        num_draws_str = f"1 ball"
        with_replacement_str = ""
    else:
        num_draws_str = f"{row['ndraws_from_cage']} balls"
        with_replacement_str = " with replacement"

    explanation = \
f"""

This experiment involves two stages. In stage 1 we will show you some information including the result of a drawing of {num_draws_str}{with_replacement_str} from one of two possible cages, each containing different numbers of light and dark balls. Then at the start of stage 2 you will report a number P between 0 and 1. After your report, we will draw a random number U that is equally likely to be any number between 0 and 1. Your payoff from this experiment will either be $1000 or $0 depending on your report P and the random number U.

Let's describe the two stages in more detail now. In stage 1 we will show you {num_draws_str} that are drawn at random{with_replacement_str} from one of two possible urns labelled A and B.

Urn A contains 2 light balls and 1 dark ball.
Urn B contains 1 light ball and 2 dark balls. 

We select the urn, A or B, from which we draw the sample of {num_draws_str} by the outcome of throwing a 6 sided die.
We do not show you the outcome of this throw of the die but we do tell you the rule we use to select urn A or B.

If the outcome of the die throw is 1 to {prior_A_threshold} we select urn A.
If the outcome of the die throw is {prior_A_threshold+1} to 6, we use urn B to draw the random sample of {num_draws_str}{with_replacement_str}.

Once you see the outcome of the sample of {num_draws_str}, stage 1 is over and stage 2 begins.

At the start of stage 2 we ask you to report a number P between 0 and 1. Your payoff from this experiment depend on another random number, which we call U, which we draw after you report the number P. We draw the random number U in a way that every possible number between 0 and 1 has an equal chance of being selected.

Here is how you will be paid from participating in this experiment. There are two possible cases:

Case 1. If the number U is less than or equal to P then you will receive $1000 if the sample of {num_draws_str} we showed you in stage 1 was from urn A and $0 otherwise.
Case 2. If the number U is between the number P you report and 1, you will receive $1000 with probability equal to the realized value of U, but with probability 1-U you will get $0.

OK, this is the setup. Let's now start begin this experiment, starting with stage 1.

We have tossed the die (the outcome we don't show to you) and selected one of these urns according to the rule given above (i.e. urn A if the die throw was 1 to {prior_A_threshold}, and urn B otherwise). We have drawn {num_draws_str}{with_replacement_str} from the selected urn and the outcome is {', '.join(list(row['outcome']))}, i.e., {row['outcome_expand']}.

Now, we are at stage 2 where we are asking you, given the information from stage 1 to report a number P between 0 and 1 that in conjunction with the random number U will determine if you get either $1000 or $0 according to the rule given in cases 1 and 2 above.

Please report a number P between 0 and 1 that maximizes your probability of winning $1000 in this experiment.
"""
    
    # Instructions
    if row['instruction'] == "reasoning":
        instruction = \
"""
YOU ARE WELCOME TO ALSO DESCRIBE YOUR REASONING, BROKEN INTO SEPARATE STEPS, TO EXPLAIN HOW YOU ARRIVED AT YOUR FINAL ANSWER P. 
Please state your answer in the following format at the end.
Final answer: [your P value here].
"""

    elif row['instruction'] == "no reasoning":
        instruction = \
"""
PLEASE JUST REPORT P AND DO NOT PROVIDE ANY REASONING AS TO HOW YOU ARRIVED AT THE VALUE P. 
Please state your answer in the following format.
Final answer: [your P value here].
"""
    return introduction + explanation + instruction

