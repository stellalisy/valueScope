dimensions = ["length", "sarcasm", "formality", "supportiveness", "toxicity"]
prompt_template = {
    "system": "You are a helpful assistant tasked to help a user rewrite a post on Reddit based on the given requirements. The type of text you should write should be online forum post, aka Reddit-style. The writing level is average, and can have some degree of human errors. Your goal is to follow instructions to transfer the style of the comment but not the content. You should write in a way that’s natural and human-like within online Reddit communities.“",
    "requirements": "Requirements: Re-write the following reddit comment to make it \"{}\" in the context of the reddit post title. The rewrite should express the same meaning as the original comment except for the level of {}.\n\n",
    "system_rate": "You are a helpful assistant tasked to help a user rate a comment on Reddit based on the given rating scales and context. If a comment does not have enough information to be rated, simply return \"NO RATING\".",
    "length": {
        "less": "Re-write the following reddit comment to make it more concise. The rewrite should express the same meaning and tone as the original comment except for length. DO NOT change the formality of the comment, keep it casual, with Reddit style, human-like.",
        "more": "Re-write the following reddit comment to make it more lengthy. The rewrite should express the same meaning and tone as the original comment except for length. DO NOT change the formality of the comment, keep it casual, with Reddit style, human-like.",
    },
    "sarcasm": {
        "less": "Re-write the following reddit comment to make it less sarcastic. The rewrite should express the same meaning as the original comment except for the level of sarcasm, keep it in Reddit style and human-like.",
        "more": "Re-write the following reddit comment to make it more sarcastic. The rewrite should express the same meaning as the original comment except for the level of sarcasm, keep it in Reddit style and human-like.",
    },
    "formality": {
        "less": "Re-write the following reddit comment to make it less formal. The rewrite should express the same meaning as the original comment except for the level of formality, keep it in Reddit style and human-like.",
        "more": "Re-write the following reddit comment to make it more formal. The rewrite should express the same meaning as the original comment except for the level of formality, keep it in Reddit style and human-like.",
    },
    "supportiveness": {
        "less": "Re-write the following reddit comment to make it less supportive. The rewrite should express the same meaning as the original comment except for the level of supportiveness, keep it in Reddit style and human-like.",
        "more": "Re-write the following reddit comment to make it more supportive. The rewrite should express the same meaning as the original comment except for the level of supportiveness, keep it in Reddit style and human-like.",
    },
    "toxicity": {
        "less": "Re-write the following reddit comment to make it less toxic. The rewrite should express the same meaning as the original comment except for the level of toxicity, keep it in Reddit style and human-like.",
        "more": "Re-write the following reddit comment to make it more toxic. The rewrite should express the same meaning as the original comment except for the level of toxicity, keep it in Reddit style and human-like.",
    },
    "nothing": "\nIn your response, generate the rewritten comment and NOTHING ELSE."
}

llama2_prompt = {
    "system": """<s>[INST] <<SYS>>\nInstruction: You are a helpful assistant tasked to help a user rewrite a post on Reddit based on the given requirements and rating scales. If a comment does not have enough information to be rewritten, simply return the original comment. Return the rewritten comment and NOTHING ELSE.\n<</SYS>>\n\n""",
    "length": {
        "less": "Re-write the following reddit comment to make it more concise. The rewrite should express the same meaning and tone as the original comment except for length. DO NOT change the formality of the comment.",
        "more": "Re-write the following reddit comment to make it more lengthy. The rewrite should express the same meaning and tone as the original comment except for length. DO NOT change the formality of the comment, keep it casual.",
    },
    "sarcasm": {
        "less": "Re-write the following reddit comment to make it less sarcastic. The rewrite should express the same meaning as the original comment except for the level of sarcasm.",
        "more": "Re-write the following reddit comment to make it more sarcastic. The rewrite should express the same meaning as the original comment except for the level of sarcasm.",
    },
    "formality": {
        "less": "Re-write the following reddit comment to make it less formal. The rewrite should express the same meaning as the original comment except for the level of formality.",
        "more": "Re-write the following reddit comment to make it more formal. The rewrite should express the same meaning as the original comment except for the level of formality.",
    },
    "supportiveness": {
        "less": "Re-write the following reddit comment to make it less supportive. The rewrite should express the same meaning as the original comment except for the level of supportiveness.",
        "more": "Re-write the following reddit comment to make it more supportive. The rewrite should express the same meaning as the original comment except for the level of supportiveness.",
    },
    "toxicity": {
        "less": "Re-write the following reddit comment to make it less toxic. The rewrite should express the same meaning as the original comment except for the level of toxicity.",
        "more": "Re-write the following reddit comment to make it more toxic. The rewrite should express the same meaning as the original comment except for the level of toxicity.",
    },
    "rating": {
        "only": "Return the numerical rating ONLY, NO EXPLANATIONS.[/INST]",
        "length": "[INST] Now, can you rate the length of the rewritten comment on a scale of 0-6? 0 means extremely short; 1 means moderately short; 2 means slightly short; 3 means average length; 4 means slightly long; 5 means moderately long; 6 means extremely long.",
        "sarcasm": "[INST] Now, can you rate the level of sarcasm of the rewritten comment on a scale of 0-6? 0 means not sarcastic at all; 1 means slightly sarcastic; 2 means moderately sarcastic; 3 means average level of sarcasm; 4 means slightly sarcastic; 5 means moderately sarcastic; 6 means extremely sarcastic.",
        "formality": "[INST] Now, can you rate the level of formality of the rewritten comment on a scale of 0-6? 0 means extremely informal; 1 means moderately informal; 2 means slightly informal; 3 means average level of formality; 4 means slightly formal; 5 means moderately formal; 6 means extremely formal.",
        "supportiveness": "[INST] Now, can you rate the level of supportiveness of the rewritten comment on a scale of 0-6? 0 means extremely unsupportive; 1 means moderately unsupportive; 2 means slightly unsupportive; 3 means average level of supportiveness; 4 means slightly supportive; 5 means moderately supportive; 6 means extremely supportive.",
        "toxicity": "[INST] Now, can you rate the level of toxicity of the rewritten comment on a scale of 0-6? 0 means extremely non-toxic; 1 means moderately non-toxic; 2 means slightly non-toxic; 3 means average level of toxicity; 4 means slightly toxic; 5 means moderately toxic; 6 means extremely toxic.",
    },
    "original_rating": {
        "length": "[INST] Can you rate the length of the following comment on a scale of 0-6? 0 means extremely short; 1 means moderately short; 2 means slightly short; 3 means average length; 4 means slightly long; 5 means moderately long; 6 means extremely long.",
        "sarcasm": "[INST] Can you rate the level of sarcasm of the following comment on a scale of 0-6? 0 means not sarcastic at all; 1 means slightly sarcastic; 2 means moderately sarcastic; 3 means average level of sarcasm; 4 means slightly sarcastic; 5 means moderately sarcastic; 6 means extremely sarcastic.",
        "formality": "[INST] Can you rate the level of formality of the following comment on a scale of 0-6? 0 means extremely informal; 1 means moderately informal; 2 means slightly informal; 3 means average level of formality; 4 means slightly formal; 5 means moderately formal; 6 means extremely formal.",
        "supportiveness": "[INST] Can you rate the level of supportiveness of the following comment on a scale of 0-6? 0 means extremely unsupportive; 1 means moderately unsupportive; 2 means slightly unsupportive; 3 means average level of supportiveness; 4 means slightly supportive; 5 means moderately supportive; 6 means extremely supportive.",
        "toxicity": "[INST] Can you rate the level of toxicity of the following comment on a scale of 0-6? 0 means extremely non-toxic; 1 means moderately non-toxic; 2 means slightly non-toxic; 3 means average level of toxicity; 4 means slightly toxic; 5 means moderately toxic; 6 means extremely toxic.",
        "only": "Return the numerical rating ONLY, NO EXPLANATIONS.",
    }
}

llama2_rating_prompt = {
    "system": """<s>[INST] <<SYS>>\nInstruction: You are a helpful assistant tasked to help a user rewrite a post on Reddit based on the given requirements and rating scales. If a comment does not have enough information to be rewritten, simply return the original comment. Return the rewritten comment and NOTHING ELSE.\n<</SYS>>\n\n""",

    "system_rate": """<s>[INST] <<SYS>>\nInstruction: You are a helpful assistant tasked to help a user rate a comment on Reddit based on the given rating scales and context. If a comment does not have enough information to be rated, simply return "NO RATING". Return the rating and NOTHING ELSE.\n<</SYS>>\n\n""",
    
    "definitions": {

        "length": """1. "Very Short": contains brief and minimal amount of text, ranging from a few words to a single short sentence
2. "Somewhat Short": presents a concise amount of information, typically spanning a few short sentences to a single, verbose sentence. 
3. "Neutral": strikes a balance between brevity and thoroughness, usually around a paragraph in length. Ranges from a couple longer sentences to multiple shorter sentences. 
4. "Somewhat Long": contains substantial amount of information, such as explanations and reasoning. Typically around 2-3 paragraphs long, with each paragraph consisting of several shorter sentences to few longer sentences. 
5. "Very Long": encompasses comprehensive and detailed information, often organized into multiple sections, accompanied by examples and thorough explanations. Typically around 4+ paragraphs.\n\n""",

        "formality": """1. "Very Casual": extensive use of abbreviations, slangs, non-standard capitalization, missing syntactic components (no noun, no verb in sentence), incorrect punctuations, colloquialisms, contractions, inappropriate language (e.g. cuss words).
2. "Somewhat Casual": existence of slangs, missing syntactic components (no noun, no verb in sentence), unnecessary use of exclamation marks, inappropriate language (e.g. cuss words, “idiots”), or persistent presence of nonstandard capitalization, missing/incorrect punctuations, abbreviations, colloquialisms, contractions, nonstandard grammar and spelling. 
3. "Neutral": Presence of a few nonstandard capitalization (e.g. not capitalized first letter of sentence), missing/incorrect punctuation, nonstandard grammar and spelling, abbreviation, colloquialisms, and relatively complete sentences. No slangs or emojis. 
4. "Somewhat Formal": syntactically well structured, correct capitalization, complete sentences, correct punctuation, correct grammar. No abbreviations, no slang, no colloquialisms, can have acronyms and contractions. Ex. “I appreciate it. Thank you.”
5. "Very Formal": very structured thoughts and professional language, no abbreviations/slang/contractions/colloquialisms, grammatically correct. Contains structure in terms of the content (topic sentence, explanation, reasoning, etc). Ex. “I appreciate your guidance _____ *insert details*”\n\n""",

        "supportiveness": """1. "Very Unsupportive": Aggressive, attacking the OP or others. Extremely rude, unreasonable, or even psycho. Outright judging that others are wrong/inferior. Using extremely inappropriate language.
2. "Somewhat Unsupportive": rude, unfriendly, disrespectful, promotes toxic behavior, leads to negative atmosphere. Will make a (normal) reader a little uncomfortable. Using inappropriate language.
3. "Neutral": neither supportive or toxic. Usually short texts like “Coffee and music” which doesn’t include any supportiveness or toxicity features
4. "Somewhat Supportive": respectful, constructive comments that have a positive outlook, not necessarily zealously supportive. Usually the commentator makes an effort to answer the question.
5. "Very Supportive": extremely positive, encouraging, promotes supportive & uplifting discussion. (e.g. omg i absolutely love this!!!!!)\n\n""",

        "sarcasm": """1. "Very Genuine": extremely sincere, honest, no implications. Profound or heartfelt messages.
2. "Somewhat Genuine": sincere and authentic, not lying. Includes subjective opinions that have enough content and context to judge as genuine (i.e. not a few words). E.g. some helpful advice.
3. "Neutral": Neither genuine nor sarcastic. Often includes short, objective answers (i.e. 1-3 words) that don’t imply anything. 
4. "Somewhat Sarcastic": appears nice, but actual meaning is opposite to textual meaning and is often negative. Often an intention to be funny. 
5. "Very Sarcastic": extreme ridicule or mockery, implicitly insulting. Exaggerated verbal irony.\n\n""",

        "politeness": """1. "Very Rude": disrespectful, demanding, offensive tone. E.g. “get the fuck out, shut up.”
2. "Somewhat Rude": not considering others feelings, imposing, generalizing without knowing the full context. E.g. judgy: “people like you would never…”, giving unsolicited advice: “Never …!” or comments that don’t really answer the question. Using exclamation/all caps when unnecessary. Often does not save their own or other’s face. 
3. "Neutral": neither showing concern for others’ “face” nor being disrespectful. E.g. “you can do this…,”. Often includes comments that are straightforward but not rude. “bald-on record politeness” in politeness theory.
4. "Somewhat Polite": Making individuals feel good about themselves (appealing to positive face) or making the individuals feel like they haven’t been imposed upon/taken advantage of (appealing to negative face). in case of agreement: friendliness and camaraderie, compliments, common grounds; in case of disagreeing opinions: not assuming, not coercing, recognizing and addressing the hearer's right to make his or her own decisions freely. (E.g. No offense but…, People usually…, I’m sure you know more than I do but…, replacing “I” and “you” with “people” or “we”). “positive politeness” and “negative politeness” in politeness theory.
5. "Very Polite": showing concern for others. give hints, give clues of association, presuppose, understate, overstate, use tautologies. Rely on the hearer to understand implications (e.g. I would do…, do you think you want to…) “Off-record politeness” in politeness theory.""",

        "humor": """1. "Very Serious": language and tone indicative of solemnity or earnestness, with a focus on conveying information or opinions with gravity and sincerity. Look for expressions of concern, absence of humor, and a straightforward communication style.
2. "Somewhat Serious": maintains a moderate level of seriousness, can include a mix of formal and informal language, occasional expressions of concern, and a balance between conveying important information or opinions with some degree of approachability.
3. "Neutral": not trying to be serious or humorous, or striking a balance between seriousness and humor. includes neutral expressions, and a versatile communication style adaptable to the context.
4. "Somewhat Humorous": incorporates humor or light-hearted language in a manner that enhances the discussion without detracting from its overall message. Can include humorous anecdotes, and playful expressions that contribute positively to the conversation.
5. "Very Humorous": primarily focuses on humor and entertainment, with language and expressions intended to amuse other users. Include witty remarks and humorous anecdotes that prioritize laughter and enjoyment over seriousness."""
    }
}