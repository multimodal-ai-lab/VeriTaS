# Instructions
You are given a claim. **Your task is to decide whether the claim is _shareable_.**

This can be determined by answering the following question: _Would any reasonable person/organization ever publicly share this claim as a standalone statement on social media?_ If the answer is no, the claim is unshareable.

Consider especially length, specificity, relevance, value, and understandability. An overly lengthy, specific, or complicated claim is rather unshareable. Same for claims that are absolutely irrelevant, uninteresting, or not worth sharing. Claims lacking context are also unshareable. 

The overall goal is to make sure that the claim stylistically matches the schema of typical claims picked up and reproduced by fact-checkers.

# Required Output Format
1. **Reasoning (one paragraph only)**: Discuss step by step whether the claim is shareable or not.
2. **Final Decision**: Answer with `shareable` or `unshareable`, accordingly. Return the decision enclosed in single backticks.

# The Claim We Are Interested In
{claim}

# Examples
**Claim**: The video shows a wife assaulting her husband and his female friend in a gym.
**Decision**: `shareable` because it is concise and shows a viral scene.

**Claim**: The video showing multiple bikes slipping and causing accidents on a rain-soaked flyover was recorded on Rashid Minhas Road in Karachi, Pakistan.
**Decision**: `unshareable` because it is too specific ("Rashid Minhas Road") and the video description is unnecessary to get the claim.

**Claim**: The story that rapper Lil Baby was found shot dead in a parked car is a prank.
**Decision**: `shareable` because it can be seen as a legitimate correction of a viral fake story.

**Claim**: Canadian COVID-19 data and expert assessments show that people vaccinated with an additional COVID-19 dose are less likely to die of COVID-19 than unvaccinated people and do not develop AIDS from vaccination.
**Decision**: `unshareable` way too specific and it mixes two wildly different things (COVID-19 and AIDS).

**Claim**: British Airways is restructuring and laying off up to 12,000 employees.
**Decision**: `shareable` because this information is of high public interest and brief.

**Claim**: The story that U.S. Navy SEALs rescued more than 1,000 trafficked children and recovered dead bodies from shipping containers on Evergreen’s Ever Given after it was stuck in the Suez Canal is fake.
**Decision**: `unshareable` because it negates an event that is very implausible anyway. So the claim has no value.

# Your Response