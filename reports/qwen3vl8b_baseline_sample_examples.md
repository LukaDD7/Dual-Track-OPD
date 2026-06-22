# Qwen3-VL-8B Baseline Raw Example Samples

These examples are sampled from raw responses for qualitative inspection. Do not commit large raw JSONL files.

## BLINK / val_Relative_Depth_1

- file: `qwen3vl8b_BLINK_len32768_maxtok128_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?
Select from the following choices.
(A) A is closer
(B) B is closer

**Options**

(not found in recognized fields)

**Prediction**

(B) B is closer

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["(B)"]

## BLINK / val_Relative_Depth_2

- file: `qwen3vl8b_BLINK_len32768_maxtok128_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?
Select from the following choices.
(A) A is closer
(B) B is closer

**Options**

(not found in recognized fields)

**Prediction**

(B) B is closer

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["(B)"]

## DynaMath_Sample / 1

- file: `qwen3vl8b_DynaMath_Sample_len32768_maxtok2048_raw.jsonl`
- row_index: `1`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

What is the period of this function y = a*sin(b*x), where both a and b are integers? Answer the question with a floating-point number.

Answer briefly. If numeric, give only the final value.

**Options**

(not found in recognized fields)

**Prediction**

3.141592653589793

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["2.0944"]

## DynaMath_Sample / 2

- file: `qwen3vl8b_DynaMath_Sample_len32768_maxtok2048_raw.jsonl`
- row_index: `2`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Use the graph to answer the question below. Which month is the hottest on average in Cape Town?

Answer briefly. If numeric, give only the final value.

**Options**

(not found in recognized fields)

**Prediction**

Sep

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["Sep"]

## GQA / 05515938

- file: `qwen3vl8b_GQA_val_balanced_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `1`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What is this bird called?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

cockatoo

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["parrot"]

## GQA / 17197213

- file: `qwen3vl8b_GQA_val_balanced_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `2`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What color is the helmet in the middle of the image?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

blue

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["light blue"]

## MMBench / 241

- file: `qwen3vl8b_MMBench_dev_len65536_maxtok256_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

The passage below describes an experiment. Read the passage and then follow the instructions below.

Madelyn applied a thin layer of wax to the underside of her snowboard and rode the board straight down a hill. Then, she removed the wax and rode the snowboard straight down the hill again. She repeated the rides four more times, alternating whether she rode with a thin layer of wax on the board or not. Her friend Tucker timed each ride. Madelyn and Tucker calculated the average time it took to slide straight down the hill on the snowboard with wax compared to the average time on the snowboard without wax.
Figure: snowboarding down a hill.

Identify the question that Madelyn and Tucker's experiment can best answer.

Options:
A. Does Madelyn's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?
B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MMBench / 252

- file: `qwen3vl8b_MMBench_dev_len65536_maxtok256_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

People can use the engineering-design process to develop solutions to problems. One step in the process is testing if a potential solution meets the requirements of the design.
The passage below describes how the engineering-design process was used to test a solution to a problem. Read the passage. Then answer the question below.

Laura and Isabella were making batches of concrete for a construction project. To make the concrete, they mixed together dry cement powder, gravel, and water. Then, they checked if each batch was firm enough using a test called a slump test.
They poured some of the fresh concrete into an upside-down metal cone. They left the concrete in the metal cone for 30 seconds. Then, they lifted the cone to see if the concrete stayed in a cone shape or if it collapsed. If the concrete in a batch collapsed, they would know the batch should not be used.
Figure: preparing a concrete slump test.

Which of the following could Laura and Isabella's test show?

Options:
A. if the concrete from each batch took the same amount of time to dry
B. if a new batch of concrete was firm enough to use

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MMMU_Pro_10 / test_History_1

- file: `qwen3vl8b_MMMU_Pro_10options_len65536_maxtok256_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Which of the following best explains the overall trend shown in the <image 1>?

Options:
['Political instability leading to population decline', 'The spread of pathogens across the Silk Road', 'Development of new trade routes', 'Climate change affecting the Silk Road', 'Migrations to areas of Central Asia for resettlement', 'Technological advancements in transportation', 'Invasions by Mongol tribes', 'Large-scale famine due to crop failures', 'Economic prosperity and population growth', 'Rise of religious conflicts along the Silk Road']

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

Economic prosperity and population growth

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MMMU_Pro_10 / test_Art_113

- file: `qwen3vl8b_MMMU_Pro_10options_len65536_maxtok256_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

<image 1> of Louis Black, believed to be a formerly enslaved man, was painted by which artist?

Options:
['James Irvine', 'John Duncan', 'Gavin Hamilton', 'David Gauld', 'Arthur Melville', 'Francis Cadell', 'Mary Cameron', 'John Lavery', 'Samuel Peploe', 'William York Macgregor']

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

Arthur Melville

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["A"]

## MMMU_Pro_4 / test_History_1

- file: `qwen3vl8b_MMMU_Pro_4options_len65536_maxtok256_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Which of the following best explains the overall trend shown in the <image 1>?

Options:
['Migrations to areas of Central Asia for resettlement', 'The spread of pathogens across the Silk Road', 'Invasions by Mongol tribes', 'Large-scale famine due to crop failures']

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MMMU_Pro_4 / test_Art_113

- file: `qwen3vl8b_MMMU_Pro_4options_len65536_maxtok256_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

<image 1> of Louis Black, believed to be a formerly enslaved man, was painted by which artist?

Options:
['Mary Cameron', 'Gavin Hamilton', 'James Irvine', 'Arthur Melville']

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

A

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["C"]

## MMSI-Bench / 0

- file: `qwen3vl8b_MMSI_Bench_len32768_maxtok256_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

The images are taken continuously from a first-person perspective. In which direction are you moving?
Options: A: Left while moving backward, B: Forward to the left, C: Forward to the right, D: Right while moving backward

Answer with the option letter.

**Options**

(not found in recognized fields)

**Prediction**

D

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["C"]

## MMSI-Bench / 1

- file: `qwen3vl8b_MMSI_Bench_len32768_maxtok256_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

The images are taken continuously from a first-person perspective. In which direction is the camera rotating?
Options: A: Back, B: Left, C: Right, D: Forward

Answer with the option letter.

**Options**

(not found in recognized fields)

**Prediction**

C

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MMVet / v1_0

- file: `qwen3vl8b_MMVet_len32768_maxtok2048_raw.jsonl`
- row_index: `1`
- scoring_type: `needs_judge`
- finish_reason: `stop`
- image: ``

**Question**

What is x in the equation?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

-1 or -5

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["-1<AND>-5"]

## MMVet / v1_1

- file: `qwen3vl8b_MMVet_len32768_maxtok2048_raw.jsonl`
- row_index: `2`
- scoring_type: `needs_judge`
- finish_reason: `stop`
- image: ``

**Question**

What is d in the last equation?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

d = 0.5

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["1.25<OR>=1.25<OR>5/4"]

## MV-MATH / 1

- file: `qwen3vl8b_MV_MATH_len65536_maxtok1024_raw.jsonl`
- row_index: `1`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

In the given solid shapes, which can be obtained by rotating a plane figure around a certain straight line for a full revolution?
A.

<image_1>
B.

<image_2>
C.

<image_3>
D.

<image_4>

Answer with the option letter.

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MV-MATH / 2

- file: `qwen3vl8b_MV_MATH_len65536_maxtok1024_raw.jsonl`
- row_index: `2`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Which of the following plane figures, when folded, cannot form a cube?

A.

<image_1>
B.

<image_2>
C.

<image_3>
D.

<image_4>

Answer with the option letter.

**Options**

(not found in recognized fields)

**Prediction**

A

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["C"]

## MathVerse / 1

- file: `qwen3vl8b_MathVerse_len65536_maxtok1024_raw.jsonl`
- row_index: `1`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.
Question: As shown in the figure, in triangle ABC, it is known that angle A = 80.0, angle B = 60.0, point D is on AB and point E is on AC, DE parallel BC, then the size of angle CED is ()
Choices:
A:40°
B:60°
C:120°
D:140°

**Options**

(not found in recognized fields)

**Prediction**

C

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["D"]

## MathVerse / 2

- file: `qwen3vl8b_MathVerse_len65536_maxtok1024_raw.jsonl`
- row_index: `2`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.
Question: As shown in the figure, it is known that angle A = 80.0, angle B = 60.0, DE parallel BC, then the size of angle CED is ()
Choices:
A:40°
B:60°
C:120°
D:140°

**Options**

(not found in recognized fields)

**Prediction**

C

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["D"]

## MathVista / 1

- file: `qwen3vl8b_MathVista_len32768_maxtok2048_raw.jsonl`
- row_index: `1`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

When a spring does work on an object, we cannot find the work by simply multiplying the spring force by the object's displacement. The reason is that there is no one value for the force-it changes. However, we can split the displacement up into an infinite number of tiny parts and then approximate the force in each as being constant. Integration sums the work done in all those parts. Here we use the generic result of the integration.

In Figure, a cumin canister of mass $m=0.40 \mathrm{~kg}$ slides across a horizontal frictionless counter with speed $v=0.50 \mathrm{~m} / \mathrm{s}$. It then runs into and compresses a spring of spring constant $k=750 \mathrm{~N} / \mathrm{m}$. When the canister is momentarily stopped by the spring, by what distance $d$ is the spring compressed?

Return only the final numeric value rounded to 1 decimal place(s). Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

0.023

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["1.2"]

## MathVista / 2

- file: `qwen3vl8b_MathVista_len32768_maxtok2048_raw.jsonl`
- row_index: `2`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

what is the total volume of the measuring cup?

Return only the final integer. Do not provide reasoning. Do not include the unit; the unit is g.

**Options**

(not found in recognized fields)

**Prediction**

1000

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["1000"]

## MindCube-Bench / among_group002_q0_1_1

- file: `qwen3vl8b_MindCube_Bench_len65536_maxtok256_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Based on these two views showing the same scene: in which direction did I move from the first view to the second view? A. Directly left B. Diagonally forward and right C. Diagonally forward and left D. Directly right

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

D

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["C"]

## MindCube-Bench / among_group002_q0_1_2

- file: `qwen3vl8b_MindCube_Bench_len65536_maxtok256_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Based on these two views showing the same scene: in which direction did I move from the first view to the second view? A. Diagonally forward and left B. Diagonally forward and right C. Directly left D. Directly right

Return only the final option letter.

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## ReMI / 0

- file: `qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl`
- row_index: `1`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Let $h(x)=f(x)$ for $x\in(-\infty, 1]$ and $h(x)=g(x)$ for $x\in(1, \infty)$. The graph of $f(x)$ is as follows <image1> and $g(x)$ as follows <image2>. Compute $\lim_{x \to 1^+} h(x)$.

Return only the final answer. Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

2

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["1"]

## ReMI / 1

- file: `qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl`
- row_index: `2`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Here are four images. The first image is image A <image1>. The second image is image B <image2>.
    The third image is image C <image3> and the fourth image is image D <image4>.

    All images represent the same grid map of road network with points of interest
    that include stop signs, traffic lights, coffee shops, gas stations, and bus stops.
    A block represents the distance between two intersections.
    A bus stop is represented by a blue square with a logo of a bus;
    a stop sign is shown using an icon of a stop sign;
    a coffee place is shown using a orange pin with the logo of a coffee cup;
    a shopping center is represented by a blue pin with the logo of a shopping cart;
    a traffic light is shown by three color dots;
    a gas station is shown using a purple pin with the logo of a gas pump.

    The street names are shown next to the horizontal and vertical lines.
    In both the images the horizontal streets are named from top to bottom as
    E street, L street, B street, Q street, O street.
    Similarly the vertical streets are named from left to right as
    G street, C street, R street, A street, D street.

    At exactly one of the intersections there is a green pin indicating the starting location.
    At exactly one of the intersections there is a red pin indicating the ending location.

    Here is a set of directions describing a path from the starting location to the ending location.

    Directions: Head north on C street.
Turn right onto Q street.
Turn left onto R street.
Turn right onto L street.
Continue for 1 block to arrive at your destination.


    In each image there is a partial (incomplete) path outlined in the color blue. Exactly two of the
    four images have the property that their partial paths when combined together match the path that is
    described in the directions above. Please identify the pair of images. 
    The answer is simply a pair of names (e.g. A,B or C,D and so on).

    

Return only the final answer. Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

A,C

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["D,B"]

## ScienceQA-IMG / 0

- file: `qwen3vl8b_ScienceQA_IMG_len32768_maxtok128_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Which animal's mouth is also adapted for bottom feeding?

Options:
A. discus
B. armored catfish

Answer with the option letter and short answer.

**Options**

(not found in recognized fields)

**Prediction**

B. armored catfish

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B. armored catfish"]

## ScienceQA-IMG / 1

- file: `qwen3vl8b_ScienceQA_IMG_len32768_maxtok128_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Which of the following could Wendy's test show?

Options:
A. whether producing more insulin would help the bacteria grow faster
B. whether different types of bacteria would need different nutrients to produce insulin
C. whether she added enough nutrients to help the bacteria produce 20% more insulin

Answer with the option letter and short answer.

**Options**

(not found in recognized fields)

**Prediction**

C. whether she added enough nutrients to help the bacteria produce 20% more insulin

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["C. whether she added enough nutrients to help the bacteria produce 20% more insulin"]

## VQAv2 / 262148000

- file: `qwen3vl8b_VQAv2_val_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `1`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Where is he looking?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

down

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["down"]

## VQAv2 / 262148001

- file: `qwen3vl8b_VQAv2_val_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `2`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What are the people in the background doing?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

watching

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["watching"]

## ViewSpatial-Bench / 0

- file: `qwen3vl8b_ViewSpatial_Bench_len65536_maxtok128_raw.jsonl`
- row_index: `1`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Could you tell me the location of the counter in comparison to the refrigerator?

Choices:
A. right
B. front-up
C. back-left
D. front

Answer with the option letter and answer.

**Options**

(not found in recognized fields)

**Prediction**

A. right

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["A. right"]

## ViewSpatial-Bench / 1

- file: `qwen3vl8b_ViewSpatial_Bench_len65536_maxtok128_raw.jsonl`
- row_index: `2`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

How is the cabinet positioned with respect to the table?

Choices:
A. left
B. front-left
C. above-left
D. back

Answer with the option letter and answer.

**Options**

(not found in recognized fields)

**Prediction**

A. left

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["D. back"]
