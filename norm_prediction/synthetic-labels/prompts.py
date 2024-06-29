prompt_system = "You are a linguistic expert who is tasked with identifying and confirming linguistic features present in Reddit comments."

prompt_system_pairwise = "You are a linguistic expert tasked with comparing which linguistic dimension is more present between two Reddit comments."

dimensions = ["SHORT-LONG", "CASUAL-FORMAL", "SUPPORTIVE-TOXIC", "GENUINE-SARCASM", "RUDE-POLITE", "HUMOR-SERIOUS"]

pairwise_dimension = ["MORE FORMAL (or LESS CASUAL)", "MORE SUPPORTIVE (or LESS TOXIC)", "MORE SARCASTIC (or LESS GENUINE)", "MORE POLITE (or LESS RUDE)", "MORE HUMOROUS (or LESS SERIOUS)"]

prompt_pairwise = """Between COMMENT1 and COMMENT2, please determine which comment is MORE [DIMENSION_PAIRWISE]. Only use the provided post title and post description as context. The [DIMENSION] scale is provided below to help determine which comment is MORE [DIMENSION_PAIRWISE]. 

[DIMENSION] SCALE:
[DIMENSION_TEMPLATE]

Please determine which comment below is [DIMENSION_PAIRWISE] and providing reasoning for your answer. If you think COMMENT1 is [DIMENSION_PAIRWISE], ANSWER WITH "{1}" at the beginning of your response. If you think COMMENT2 is [DIMENSION_PAIRWISE], ANSWER WITH "{2}" at the beginning of your response. If you think COMMENT1 and COMMENT2 are similar, ANSWER WITH "{1.5}". Only vote "{1.5}" in the case that both comments are near similar in the [DIMENSION] scale. 

"POST TITLE1: [TITLE1]"
"POST DESCRIPTION1: [DESCRIPTION1]"
"COMMENT1: [COMMENT1]"
"POST TITLE2: [TITLE2]"
"POST DESCRIPTION2: [DESCRIPTION2]"
"COMMENT2: [COMMENT2]"
"""

prompt_pairwise_definition = """Between COMMENT1 and COMMENT2, please determine which comment is [DIMENSION_PAIRWISE] and provide reasoning for your answer. Only use the provided post title and post description as context. The [DIMENSION] definition is provided below to help determine which comment is [DIMENSION_PAIRWISE]. 

[DIMENSION] definition: [DIMENSION_DEFINITION]

If you think COMMENT1 is [DIMENSION_PAIRWISE], ANSWER WITH "{1}" at the beginning of your response. If you think COMMENT2 is [DIMENSION_PAIRWISE], ANSWER WITH "{2}" at the beginning of your response. DO NOT FORGET TO ADD YOUR ANSWER OF {1} or {2} AND INCLUDE YOUR REASONING.

"POST TITLE1: [TITLE1]"
"POST DESCRIPTION1: [DESCRIPTION1]"
"COMMENT1: [COMMENT1]"
"POST TITLE2: [TITLE2]"
"POST DESCRIPTION2: [DESCRIPTION2]"
"COMMENT2: [COMMENT2]"
"""

prompt_pairwise_definition_few_shot = """Between COMMENT1 and COMMENT2, please determine which comment below is [DIMENSION_PAIRWISE] and provide reasoning for your answer. Only use the provided post title and post description as context. The [DIMENSION] definition is provided below to help determine which comment is [DIMENSION_PAIRWISE].

[DIMENSION] definition: [DIMENSION_DEFINITION]

[FEW-SHOT]

Now, given what you learned from the examples, if you think COMMENT1 is [DIMENSION_PAIRWISE], ANSWER WITH "{1}" at the beginning of your response. If you think COMMENT2 is [DIMENSION_PAIRWISE], ANSWER WITH "{2}" at the beginning of your response. 

"POST TITLE1: [TITLE1]"
"POST DESCRIPTION1: [DESCRIPTION1]"
"COMMENT1: [COMMENT1]"
"POST TITLE2: [TITLE2]"
"POST DESCRIPTION2: [DESCRIPTION2]"
"COMMENT2: [COMMENT2]"
"""

few_shot_pairwise_formality = """We provide three examples of the task, each featuring two sets of comments alongside their answer and corresponding reasoning.

Example 1: 
EXAMPLE1_COMMENT1: "You poor kid. Anything that works is fine. Toilet paper is a fine solution."
EXAMPLE1_COMMENT2: "none, tbh. maybe increased pooping, but that could very well be coincidental."
EXAMPLE1_ANSWER: "{1}. EXAMPLE1_COMMENT1 exhibits a more formal tone compared to EXAMPLE1_COMMENT2. EXAMPLE1_COMMENT1 maintains a structured approach, using relatively complete sentences, standard capitalization, and correct punctuations. Meanwhile, EXAMPLE1_COMMENT1 is much more casual, using abbreviations (i.e. "tbh") and consistently lacking syntatic components."

Example 2: 
EXAMPLE2_COMMENT1: "Graduating college. Jeez, 25 years old."
EXAMPLE2_COMMENT2: "I get diarrhea.  I calm down in the bathroom."
EXAMPLE2_ANSWER: "{2}. EXAMPLE2_COMMENT2 is more formal compared to EXAMPLE2_COMMENT1. EXAMPLE2_COMMENT2 maintains a structured approach, utilizing complete sentences, standard capitalization, standard puncutation, and correct grammar. Additionally, EXAMPLE2_COMMENT2 also avoids the usage of abbreviations, slangs, and contractions. Meanwhile, EXAMPLE2_COMMENT1 lacks complete sentences, misses syntactic components, and uses colloquialism (i.e. "jeez")."

Example 3: 
EXAMPLE3_COMMENT1: "This will sound stupid, but I never really ate buttered toast until a month or two ago and now I *can’t stop.* god it’s so delicious"
EXAMPLE3_COMMENT2: "Baby, you’re the best a man can get!"
EXAMPLE3_ANSWER: "{2}. EXAMPLE3_COMMENT2 maintains a more formal tone compared to EXAMPLE3_COMMENT1. EXAMPLE3_COMMENT2 contains all syntactic components and is a complete sentence with standard capitalization, grammar, and punctuation. EXAMPLE3_COMMENT1 uses nonstandard capitalization and nonstandard components such as "*" and "god", thus creating a more casual tone."
"""

few_shot_pairwise_supportive = """We provide three examples of the task, each featuring two sets of comments alongside their answer and corresponding reasoning.

Example 1: 
EXAMPLE1_COMMENT1: "Focus on basic self care and creating some basic comfort for yourself."
EXAMPLE1_COMMENT2: "Connective."
EXAMPLE1_ANSWER: "{1}. EXAMPLE1_COMMENT1 is more supportive compared to EXAMPLE1_COMMENT2. EXAMPLE1_COMMENT1 offers a respectful and positive advice for other people in the community who are burnt out, whereas EXAMPLE1_COMMENT2 offers a straightforward response that addresses the post."

Example 2:
EXAMPLE2_COMMENT1: "R.I.P."
EXAMPLE2_COMMENT2: "1 year"
EXAMPLE2_ANSWER: "{2}. EXAMPLE2_COMMENT2 is relatively more supportive compared to EXAMPLE2_COMMENT1. EXAMPLE2_COMMENT2 contains a straightforward, neutral message regarding the post. However, the expression of EXAMPLE2_COMMENT1 "R.I.P." promotes sarcastic and potentially dismissive content, which can be perceived as insensitive and mocking towards one's ex and thus do not promote a constructive discussion."

Example 3: 
EXAMPLE3_COMMENT1: "Although this question is cringe, I'd say scotch/rum and (club) soda. The bite of the alcohol is taken away by the soda, so it tastes refreshing and you can appreciate the flavours without looking like you're trying too hard by ordering it neat."
EXAMPLE3_COMMENT2: "keep 10 invest the other 90"
EXAMPLE3_ANSWER: "{2}. EXAMPLE3_COMMENT2 is relatively more supportive compared to EXAMPLE3_COMMENT1. EXAMPLE3_COMMENT2 offers a straightforward response that can build on discussions and seems to make an effort to address the post. Although EXAMPLE3_COMMENT1 provides a constructive advice and seems to make an effort to address the post, it comes off dismissive and rude by calling the question "cringe", thus being slightly toxic."
"""


few_shot_pairwise_genuine = """We provide three examples of the task, each featuring two sets of comments alongside their answer and corresponding reasoning.

Example 1: 
EXAMPLE1_COMMENT1: "Great! Got into my college program (after being forced to sit out a year due to COVID), got my motorcycle license, and I’m in training to start a new and better-paying job in the fall! Only mishap is I’m not traveling as much as I wanted to from being busy training."
EXAMPLE1_COMMENT2: "I 💖 Oregon when it's not on fire"
EXAMPLE1_ANSWER: "{2}. EXAMPLE1_COMMENT2 is more sarcastic and less genuine compared to EXAMPLE1_COMMENT1. EXAMPLE1_COMMENT1 offers a sincere and authentic message that offers their personal successes and mishaps into their year, thus being very genuine. Meanwhile, EXAMPLE1_COMMENT2 sarcastically uses emojis and exaggerate that they only love Oregon "when it's not on fire", implicitly making fun of Oregon's history with wild fires."

Example 2:
EXAMPLE2_COMMENT1: "Everyone assuming I'm OK because I look and act OK."
EXAMPLE2_COMMENT2: "“Resist and bite” by Sabaton"
EXAMPLE2_ANSWER: "{2}. EXAMPLE2_COMMENT2 is relatively more sarcastic (or less genuine) compared to EXAMPLE2_COMMENT1. EXAMPLE2_COMMENT1 offers a honest and authentic message that reveals the vulnerability of the commentor. Meanwhile, EXAMPLE2_COMMENT2 is a straightforward answer to the post and thus is relatively less genuine compared to EXAMPLE2_COMMENT1."

Example 3: 
EXAMPLE3_COMMENT1: "Typical engineer, classing people as "engineer" or "non-engineer" ;)"
EXAMPLE3_COMMENT2: "Ex-wife turned out to be an abusive psycopath"
EXAMPLE3_ANSWER: "{1}. EXAMPLE3_COMMENT1 is more sarcastic compared to EXAMPLE3_COMMENT2. In EXAMPLE3_COMMENT1, the tone is sarcastic and slightly mocking. By labeling the poster as a "typical engineer" and pointing out the categorization of people based on career, the commentor implies criticism of the original post's framing and categorization, which can be interpreted as dismissive. Meanwhile, EXAMPLE3_COMMENT2 provides a sincere and authentic response to the question posed in the post, sharing personal experience and insight, which may resonate with others who have gone through similar experiences."
"""

few_shot_pairwise_polite = """We provide three examples of the task, each featuring two sets of comments alongside their answer and corresponding reasoning.

Example 1: 
EXAMPLE1_COMMENT1: "You poor kid. Anything that works is fine. Toilet paper is a fine solution."
EXAMPLE1_COMMENT2: "Not currently.  I used to dabble in blacksmithing though.  Also yarn spinning.  I enjoy learning traditional skills."
EXAMPLE1_ANSWER: "{2}. EXAMPLE1_COMMENT2 is more polite and less rude compared to EXAMPLE1_COMMENT1. EXAMPLE1_COMMENT1 carries a disrespectful and judgemental tone, saying "You poor kid" to the poster, which can be viewed as rude. On the other hand, EXAMPLE1_COMMENT2 is straightforward and addresses the post, which is neither rude nor polite."

Example 2:
EXAMPLE2_COMMENT1: "1 year":
EXAMPLE2_COMMENT2: "I’m a woman and I’m answering too, sorry.   I don’t find it as weird as knowing people have used my photos without consent .  Do guys feel differently about that?
EXAMPLE2_ANSWER: "{2}. EXAMPLE2_COMMENT2 is more polite compared to EXAMPLE2_COMMENT1. EXAMPLE2_COMMENT2 utilizes indirect langauges, apologizes (i.e. "sorry"), and also asks questions to other people about their opinions, which makes the comment less imposing on other and thus more polite. Meanwhile, EXAMPLE1_COMMENT1 is a straightforward answer that is neither rude nor polite."

Example 3: 
EXAMPLE3_COMMENT1: "Like physical affection? If I go more than a week I crave it really badly. Kinda depends on if I have somebody I can count on for it, it's easier to go a bit longer if I know I have somebody I'll see in like two weeks."
EXAMPLE3_COMMENT2: "Income tax is illegal, and you don\'t have to pay."\n\nCoincidentally, every single person I know who fell for this was prosecuted and fined a large sum. Idiots."
EXAMPLE3_ANSWER: "{1}. EXAMPLE3_COMMENT1 is more polite and less rude compared to EXAMPLE3_COMMENT2. EXAMPLE3_COMMENT1 contains a straightforward, personal response that addresses the post's question, which is not particularly polite nor rude. EXAMPLE3_COMMENT2 is disrespectful and offensive, calling other people "Idiots," which is indicative of a rude comment."
"""

few_shot_pairwise_humor = """We provide three examples of the task, each featuring two sets of comments alongside their answer and corresponding reasoning.

Example 1: 
EXAMPLE1_COMMENT1: "Paco rabane 1 million since i can’t get no millions in my bank account at least i can wear it"
EXAMPLE1_COMMENT2: "Focus on basic self care and creating some basic comfort for yourself."
EXAMPLE1_ANSWER: "{1}. EXAMPLE1_COMMENT1 is more humorous and less serious compared to EXAMPLE1_COMMENT2. EXAMPLE1_COMMENT is characterized by light-hearted language and witty remark, amusingly implying that they wear a cologne named "1 million" because they do not not have 1 million dollars. On the other hand, EXAMPLE1_COMMENT2 provides a earnest and serious response, focused on conveying their opinions with sincereity on a serious topic."

Example 2:
EXAMPLE2_COMMENT1: "I get diarrhea.  I calm down in the bathroom."
EXAMPLE2_COMMENT2: "I 💖 Oregon when it's not on fire"
EXAMPLE2_ANSWER: "{2}. EXAMPLE2_COMMENT2 is more humorous and less serious compared to EXAMPLE2_COMMENT1. EXAMPLE2_COMMENT2 indicates a light-hearted langauge intended to be funny, using emojis and sarcastically saying that they love Oregon when it's not on fire. Meanwhile, EXAMPLE2_COMMENT1 is a serious response that straightforwardedly and sincerely answers their personal strategy to remain calm during anxiety."

Example 3: 
EXAMPLE3_COMMENT1: "“Resist and bite” by Sabaton"
EXAMPLE3_COMMENT2: "Seeing my son be kind to people and sticking up for his friends. He also packs extra lunch to share if someone doesn’t have enough."
EXAMPLE3_ANSWER: "{1}. EXAMPLE3_COMMENT1 is more humorous and less serious compared to EXAMPLE3_COMMENT2. EXAMPLE3_COMMENT2 is relatively straightofrward and indicates a focus on conveying their personal experiences with sincerity, which better fits our definition of seriousness. Meanwhile, EXAMPLE3_COMMENT1 is simply a straightforward response that answers the question."
"""

prompt_rate = """Please rate the COMMENT, only using the POST TITLE and POST DESCRIPTION as context, on the provided [DIMENSION] SCALE.
[DIMENSION] SCALE:
[DIMENSION_TEMPLATE]
Please rate the COMMENT using the provided scale on [DIMENSION] and provide reasoning for your answer. Place rating between square brackets (i.e. []). 
"POST TITLE: [TITLE]"
"POST DESCRIPTION: [DESCRIPTION]"
"COMMENT: [COMMENT]"
"""

prompt_few_shot_rate = """Please rate the COMMENT, only using the POST TITLE and POST DESCRIPTION as context, on the provided [DIMENSION] SCALE.
[DIMENSION] SCALE:
[DIMENSION_TEMPLATE]

[FEW-SHOT]
Now, given what you learned from the examples, please rate the COMMENT using the provided scale on [DIMENSION] and provide reasoning for your answer. Place rating between square brackets (i.e. []). 
"POST TITLE: [TITLE]"
"POST DESCRIPTION: [DESCRIPTION]"
"COMMENT: [COMMENT]"
"""

few_shot_rate_formality = """We provide three examples, one on each line. Each example contains the comment and the human-generated answer.
Example 1: Comment: "I stayed for my pets. By the time they passed I was in a better place mentally...I\'m glad I didn\'t go through with it or I wouldn\'t have been able to met my current partner and enjoy my current situation." Answer: "[3]. This comment scores a 3 (in-between) on the scale because the comment has relatively complete sentences, but uses nonstandard punctuations such as multiple periods. "
Example 2: Comment: "Typical engineer, classing people as "engineer" or "non-engineer"" Answer: "[2]. This comment scores a 2 (casual) on the scale because the comment is missing syntactic components (i.e. nouns)."
Example 3: Comment: "i like raspberry pie" Answer: "[1]. This comment scores a 1 (very casual) because the comment extensively uses non-standard capitalization and lacks correct punctuations."
"""

few_shot_rate_supportive = """
We provide three examples, one on each line. Each example contains the comment and the human-generated answer.
Example 1: Comment: "Someone with same job as mine , I guess we would talk about work too much !!" Answer: "[3]. This comment does not contain any supportive or toxicity features, thus fitting the definition of 3 (in-between) for the scale."
Example 2: Comment: "This came up in a recent post about being unattractive. Older women find me very cute/adorable/pretty to the point that they say things unprompted, but never women my own age. I regularly get style compliments but recently (after getting my hair styled) a cashier at a drug store said "I like your hair."\n\nBut outside of that I\'ve never had a woman my age compliment my appearance directly, so...I guess?" Answer: "[2]. This sentence would be rated as 2 (supportive) because the comment is respectful and constructive to the post, creating a positive discussion."
Example 3: Comment: "I don’t let it affect me. I don’t give them any power and am happier and healthier because of it." Answer: "[2]. This comment is constructive and has a positive outlook on deflecting negativity from other people."
"""


few_shot_rate_genuine = """
We provide three examples, one on each line. Each example contains the comment and the human-generated answer.
Example 1: Comment: "Because women are hard to figure out, because they don\'t know what words are apparently, and they seem to be under the impression that we can read minds." Answer: "[5]. This comment is rated as 5 (very sarcastic) because it employs extreme ridicule and mockery that implicitly insults women and their ability to communicate."
Example 2: Comment: "Typical engineer, classing people as "engineer" or "non-engineer" ;)" Answer: "[4]. This comment is rated as 4 (Sarcastic) because it employs verbal irony to make fun of engineers."
Example 3: Comment: "I don’t let it affect me. I don’t give them any power and am happier and healthier because of it." Answer: "[2]. This comment is rated as 2 (Genuine) because the comment is sharing genuine advice on how to deflect negativity and have a better mental health."
"""

few_shot_rate_polite = """
We provide three examples, one on each line. Each example contains the comment and the human-generated answer.
Example 1: Comment: "Seeing my son be kind to people and sticking up for his friends. He also packs extra lunch to share if someone doesn’t have enough." Answer: "[3]. This comment scores a 3 (in-different) because the comment is not disrespectful nor showing concerns for others' faces based on politeness theory."
Example 2: Comment: "Key lime!" Answer: "[3]. This comment is not disrespectful nor showing concern for any others' faces, thus the comment is annotated as 3 (in-between)."
Example 3: Comment: "Typical engineer, classing people as "engineer" or "non-engineer" ;)" Answer: "[2]. This comment is 2 (rude) because the comment is generalizing all engineers (i.e. "typical engineer") on a behavior and making an unsolicited joke."
"""

casual_formal_definition = "We define formality based on the comments' usage of abbreviations, slangs, colloquialisms, non-standard capitalizations, contractions, necessary punctuations, or opening expressions of sentences. More casual comments will use more abbreviations, slangs, non-standard capitalization, inappropriate language, colloquialisms, contractions, incorrect punctuations, and missing syntactic components, while more formal comments will be syntactically well-structured, complete sentences, correct capitalization and puncutations, and correct grammar."

casual_formal = """1. Very Casual: extensive use of abbreviations, slangs, non-standard capitalization, missing syntactic components (no noun, no verb in sentence), incorrect punctuations, colloquialisms, contractions, inappropriate language (e.g. cuss words).
2. Casual: existence of slangs, missing syntactic components (no noun, no verb in sentence), unnecessary use of exclamation marks, inappropriate language (e.g. cuss words, “idiots”), or persistent presence of nonstandard capitalization, missing/incorrect punctuations, abbreviations, colloquialisms, contractions, nonstandard grammar and spelling. 
3. In-between: Presence of a few nonstandard capitalization (e.g. not capitalized first letter of sentence), missing/incorrect punctuation, nonstandard grammar and spelling, abbreviation, colloquialisms, and relatively complete sentences. No slangs or emojis. 
4. Formal: syntactically well structured, correct capitalization, complete sentences, correct punctuation, correct grammar. No abbreviations, no slang, no colloquialisms, can have acronyms and contractions. Ex. “I appreciate it. Thank you.”
5. Very Formal: very structured thoughts and professional language, no abbreviations/slang/contractions/colloquialisms, grammatically correct. Contains structure in terms of the content (topic sentence, explanation, reasoning, etc). Ex. “I appreciate your guidance *some additional details*”
"""

supportive_toxic_definition = "We define supportive comments as one that is respectful, encouraging, or positive that promotes a constructive and/or uplifting discussion; commentor seems to make an effort to answer the question. We define comments as one that is rude, aggressive, unfriendly, disrespectful, or unreasonable that may make an average reader uncomfortable."

supportive_toxic = """1. Very Supportive: extremely positive, encouraging, promotes supportive & uplifting discussion. (e.g. omg i absolutely love this!!!!!)
2. Supportive: respectful, constructive comments that have a positive outlook, not necessarily zealously supportive. Usually the commentator makes an effort to answer the question.
3. In-between: neither supportive or toxic. Usually short texts like “Coffee and music” which doesn’t include any supportiveness or toxicity features
4. Toxic: rude, unfriendly, disrespectful, promotes toxic behavior, leads to negative atmosphere. Will make a (normal) reader a little uncomfortable. Using inappropriate language.
5. Very Toxic: Aggressive, attacking the OP or others. Extremely rude, unreasonable, or even psycho. Outright judging that others are wrong/inferior. Using extremely inappropriate language.
"""

genuine_sarcasm_definition = "We define genuine comments as one that is sincere, honest, or authentic; this often includes profound messages or some helpful advice. We define sarcastic comments as one where the actual meaning is opposite to the textual meaning; sarcastic comments are characterized by extreme ridicule, mockery, implicit insults, or exaggerated verbal irony."

genuine_sarcasm = """1. Very Genuine: extremely sincere, honest, no implications. Profound or heartfelt messages.
2. Genuine: sincere and authentic, not lying. Includes subjective opinions that have enough content and context to judge as genuine (i.e. not a few words). E.g. some helpful advice.
3. Neither/In-between: Neither genuine nor sarcastic. Often includes short, objective answers (i.e. 1-3 words) that don’t imply anything. 
4. Sarcastic: appears nice, but actual meaning is opposite to textual meaning and is often negative. Often an intention to be funny. 
5. Very Sarcastic: extreme ridicule or mockery, implicitly insulting. Exaggerated verbal irony.
"""

rude_polite_definition = "Based on Levinson and Brown's politeness theory, we define polite comments as one that makes individuals feel good about themselves (appealing to positive face), making individuals feel they haven't been imposed upon (appealing to negative face), or showing concern for others, often using hints, clues of association, understatements, or tautologies for indirect languages. We define rude comments as one that are disrespectful, demanding, or offensive; rude comments often do not consider others feelings, impose, and generalize without knowing full context (i.e. judgemental, providing unsolicited advice)."

rude_polite = """1. Very Rude: disrespectful, demanding, offensive tone. E.g. “get the fuck out, shut up.”
2. Rude: not considering others feelings, imposing, generalizing without knowing the full context. E.g. judgy: “people like you would never…”, giving unsolicited advice: “Never …!” or comments that don’t really answer the question. Using exclamation/all caps when unnecessary. Often does not save their own or other’s face. 
3. In-between: neither showing concern for others’ “face” nor being disrespectful. E.g. “you can do this…,”. Often includes comments that are straightforward but not rude. “bald-on record politeness” in politeness theory.
4. Polite: Making individuals feel good about themselves (appealing to positive face) or making the individuals feel like they haven’t been imposed upon/taken advantage of (appealing to negative face). in case of agreement: friendliness and camaraderie, compliments, common grounds; in case of disagreeing opinions: not assuming, not coercing, recognizing and addressing the hearer's right to make his or her own decisions freely. (E.g. No offense but…, People usually…, I’m sure you know more than I do but…, replacing “I” and “you” with “people” or “we”). “positive politeness” and “negative politeness” in politeness theory.
5. Very Polite: showing concern for others. give hints, give clues of association, presuppose, understate, overstate, use tautologies. Rely on the hearer to understand implications (e.g. I would do…, do you think you want to…) “Off-record politeness” in politeness theory.
"""

humor_serious_definition = "We define humorous comments as one that have light-hearted language and expressions intended to amuse others; humorous comments includes witty remarks, funny anecdotes that prioritizes laughters and enjoyment over seriousness. We define serious comments as one that indicate solemnity or earnestness with a focus on conveying information or opinions with gravity and sincereity. These comments are often characterized by absence of humor and straightforward communication style. "


humor_serious = """1. Very Serious: language and tone indicative of solemnity or earnestness, with a focus on conveying information or opinions with gravity and sincerity. Look for expressions of concern, absence of humor, and a straightforward communication style.
2. Serious: maintains a moderate level of seriousness, can include a mix of formal and informal language, occasional expressions of concern, and a balance between conveying important information or opinions with some degree of approachability.
3. In-between: not trying to be serious or funny, or striking a balance between seriousness and humor. includes neutral expressions, and a versatile communication style adaptable to the context.
4. Humorous: incorporates humor or light-hearted language in a manner that enhances the discussion without detracting from its overall message. Can include humorous anecdotes, and playful expressions that contribute positively to the conversation.
5. Very Humorous: primarily focuses on humor and entertainment, with language and expressions intended to amuse other users. Include witty remarks and humorous anecdotes that prioritize laughter and enjoyment over seriousness.
"""
