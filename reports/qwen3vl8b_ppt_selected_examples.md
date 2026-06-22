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

## BLINK / val_Visual_Correspondence_127

- file: `qwen3vl8b_BLINK_len32768_maxtok128_raw.jsonl`
- row_index: `375`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. You are given multiple red-circled points on the second image, choices of "A, B, C, D" are drawn beside each circle. Which point on the second image corresponds to the point in the first image? Select from the following options.
(A) Point A
(B) Point B
(C) Point C
(D) Point D

**Options**

(not found in recognized fields)

**Prediction**

A

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["(A)"]

## BLINK / test_Forensic_Detection_26

- file: `qwen3vl8b_BLINK_len32768_maxtok128_raw.jsonl`
- row_index: `750`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

You are a judge in a photography competition, and now you are given the four images. Please examine the details and tell which one of them is most likely to be a real photograph.
Select from the following choices.
(A) the first image
(B) the second image
(C) the third image
(D) the fourth image

**Options**

(not found in recognized fields)

**Prediction**

(C) the third image

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["hidden"]

## BLINK / test_Relative_Reflectance_134

- file: `qwen3vl8b_BLINK_len32768_maxtok128_raw.jsonl`
- row_index: `1124`
- scoring_type: `mcq`
- finish_reason: `stop`
- image: ``

**Question**

Two points are annotated on the image, labeled by A and B. Consider the surface color of the points (the albedo of the surface, without the effect of shading). Which point has darker surface color, or the colors is about the same? Select from the following choices.
(A) A is darker
(B) B is darker
(C) About the same

**Options**

(not found in recognized fields)

**Prediction**

A

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["hidden"]

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

## GQA / 03150110

- file: `qwen3vl8b_GQA_val_balanced_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `1667`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Are there any flags or kites that are not yellow?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

no

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["no"]

## GQA / 17103719

- file: `qwen3vl8b_GQA_val_balanced_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `3334`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Is the bus on the left of the picture?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

no

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["no"]

## GQA / 13356908

- file: `qwen3vl8b_GQA_val_balanced_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `5000`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What color is his shirt?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

blue

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["blue"]

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

## MathVerse / 1314

- file: `qwen3vl8b_MathVerse_len65536_maxtok1024_raw.jsonl`
- row_index: `1314`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.
Question: Triangle L M N is equilateral, and M P bisects L N. Find the measure of the side of \triangle L M N.
Choices:
A:9
B:10
C:11
D:12

**Options**

(not found in recognized fields)

**Prediction**

B

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["B"]

## MathVerse / 2627

- file: `qwen3vl8b_MathVerse_len65536_maxtok1024_raw.jsonl`
- row_index: `2627`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.
Question: What is the value of the x-coordinate of point A?
Choices:
A:$\sin \left(50^{\circ}\right)$
B:$\cos \left(50^{\circ}\right)$
C:$\sin \left(140^{\circ}\right)$
D:$\cos \left(140^{\circ}\right)$
E:$\sin \left(230^{\circ}\right)$
F:$\cos \left(230^{\circ}\right)$

**Options**

(not found in recognized fields)

**Prediction**

D

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["F"]

## MathVerse / 3940

- file: `qwen3vl8b_MathVerse_len65536_maxtok1024_raw.jsonl`
- row_index: `3940`
- scoring_type: `numeric_exact`
- finish_reason: `stop`
- image: ``

**Question**

According to the question shown in the image, please directly answer the question and provide the correct option letter, e.g., A, B, C, D.

**Options**

(not found in recognized fields)

**Prediction**

B

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

## MathVista / 334

- file: `qwen3vl8b_MathVista_len32768_maxtok2048_raw.jsonl`
- row_index: `334`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Are there fewer yellow metal tandem bikes in front of the small yellow metallic bicycle than metal bicycles on the left side of the large brown jet?

Choices:
['Yes' 'No']

Return only the final option letter. Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

Yes

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["Yes"]

## MathVista / 667

- file: `qwen3vl8b_MathVista_len32768_maxtok2048_raw.jsonl`
- row_index: `667`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Which year has the least difference between the used and new cars?

Return only the final integer. Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

2013

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["2015"]

## MathVista / 1000

- file: `qwen3vl8b_MathVista_len32768_maxtok2048_raw.jsonl`
- row_index: `1000`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Subtract all brown matte cylinders. Subtract all big purple matte things. How many objects are left?

Return only the final integer. Do not provide reasoning.

**Options**

(not found in recognized fields)

**Prediction**

9

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["9"]

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

## VQAv2 / 757014

- file: `qwen3vl8b_VQAv2_val_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `1667`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What are the elephants leaning standing in front of?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

water

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["water"]

## VQAv2 / 394840001

- file: `qwen3vl8b_VQAv2_val_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `3334`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

Is there a man on top of the horse?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

yes

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["yes"]

## VQAv2 / 2529009

- file: `qwen3vl8b_VQAv2_val_limit5000_len65536_maxtok128_raw.jsonl`
- row_index: `5000`
- scoring_type: `normalized_exact`
- finish_reason: `stop`
- image: ``

**Question**

What is the weather like?

Answer briefly.

**Options**

(not found in recognized fields)

**Prediction**

sunny

**Reasoning / Explanation**

(not found in recognized fields)

**Ground Truths**

["sunny"]
