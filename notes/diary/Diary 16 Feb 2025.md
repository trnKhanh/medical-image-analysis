---
Date created: "16-02-2025"
Time created: "14:19"
---

# Summary
- I have try human-in-the-loop scheme and achieve the highest DICE score of 0.933929 (5 folds ensemble) and 0.927067 (fold 0).
- We manually the mask of around 200 images based on the following criteria:
	- Number of samples per variant (vastly different from other variant) is from 5 to 10 depending on the model performance.
	- We mainly refine the mask by remove the noise (disconnected components), miss-labeled, non-convex mask, and rough boundary.
- We have done 2 loops:
	- 1st loop: include 88 images.
	- 2nd loop: include 200 images.
	- Note that the time taken to annotate in the second loop is faster than than the first loop because the models could learn from various samples. (The labeled data only have around 2 variant).

# TODO
- FUGC 2025:
	- Annotating the whole dataset again manually.
	- There are many parts we were not sure about the label correctness. We probably need the specialist knowledge to solve this problem and improve the performance.
- Read more about semi-supervised learning in medical domain in order to:
	- Write the report about FUGC 2025 and submit 4-page paper before 26 Feb 2025.
	- Get the overall idea and SOTA in the field. Our current scheme involves humans labor significantly (even though we do not need specialists/doctors to annotate our results). We could do it because the FUGC 2025 challenges have a relatively small dataset size (500 images = 50 labeled + 450 labeled).
