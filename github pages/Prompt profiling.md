---
title: Prompt Profiling
---
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
      extensions: ["tex2jax.js"],
      tex2jax: {
          inlineMath: [ ['$','$'], ["\\(","\\)"] ],
          processEscapes: true,
          processRefs: true,
          processEnvironments: true
      },
      TeX: { equationNumbers: { autoNumber: "AMS" } }
  });
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

**The Problem**

Large Language Models have the powerful ability to handle a huge variety of topics. Summarisation, text extraction, coding, and problem-solving are only a small subset of what they can do.

All of these tasks—despite being quite different in nature—can be achieved by the exact same model through the use of a specific set of parameters: **the prompt**.

If you've ever built features with LLMs, you already know that prompts can be tricky to tune. The high-level idea is quite easy: if I want to develop a summariser, I pass the input text alongside an instruction like _"Write a summary of the text."_ However, in production, you almost always need to add complexities that must be reflected in the prompt, such as:

- Changing the output language
    
- Following a strict output structure (like JSON)
    
- Adopting a specific syntax, tone, or behavioural constraint
    

Once you introduce these requirements, instructions become less trivial to define. Even worse, the optimal phrasing often depends entirely on the specific model you are using.

Known methods for designing prompts are often too generalised. Advice like "Use Markdown," "Use Chain of Thought," or "Add few-shot examples" doesn't help you determine the exact wording you should use, and what are the sections that you want to have, and the ones that you want not to.

What I propose here is an approach to logically profile a prompt. By breaking it down, we can identify how "good" each individual element is based on a strict metric function.

**The Solution: Prompts as Trees**

In order to achieve the best possible performance, providers recommend structuring your prompts using formats like Markdown or XML.

Consider an example system prompt for an LLM tasked with sentiment analysis, topic detection, and summarisation:

```Markdown
# Instructions
You are a text analysis engine.
Your task is to analyze the provided text and return a structured JSON object 
that strictly follows the required schema.
You must ALWAYS return valid JSON only. 
Do not include explanations, markdown, or extra text.

## OUTPUT SCHEMA
textanalysisresult
{
	// Chain of thought reasoning for your analysis
	reasoning: string,
	client_sentiment: positive | negative | neutral | mixed,
	topics: { name: string, relevance: float }[],
	key_phrases: string[],
	summary: string,
}

## RULES
- Output MUST be valid JSON.
- Do NOT include any text outside JSON.
- Do NOT wrap output in markdown.
- Always follow the schema exactly.
- If a field has no data, return an empty list or null as appropriate.
- Relevance scores must be between 0 and 1.
- Topics should reflect the main themes of the text.

## MULTILINGUAL SUPPORT
Input text can be in ANY language.
All your analysis must be in ENGLISH.
The output fields must follow these rules:
- "topics.name": must be in ENGLISH
- "key_phrases": must be extracted in the SAME language as the input text
- "summary": must be written in ENGLISH
- "client_sentiment": always one of: positive, negative, neutral, mixed (never translated)
```

The core value of my proposal is to **consider structured text like this as an Abstract Syntax Tree (AST)**.

Each section is a node. Each bulleted list is a node. All the nodes are composed of other nodes in a recursive hierarchy. The only non-composite nodes—the leaves—are the base identities. An identity is a single sentence or rule, like _"Capture main intent and outcome of the text."_

Adapting our prompt to this representation yields a tree like this:

```plaintext
		
  │-----_Section[Instructions]
  │     ├─ _Array                                         ← intro sentences
  │     │  ├─ "You are a text analysis engine."
  │     │  ├─ "Your task is to analyze the provided text…"
  │     │  ├─ "that strictly follows the required schema."
  │     │  ├─ "You must ALWAYS return valid JSON only."
  │     │  └─ "Do not include explanations, markdown…"
  │     ├─ _Section[OUTPUT SCHEMA]                        ← title node
  │     │  └─ _Section[textanalysisresult]                ← title node
  │     │     ├─ _Field[reasoning]                        ← leaf
  │     │     ├─ _Field[client_sentiment]                 ← leaf
  │     │     ├─ _Field[topics]                           ← leaf
  │     │     ├─ _Field[key_phrases]                      ← leaf
  │     │     └─ _Field[summary]                          ← leaf
  │     ├─ _Section[RULES]                                ← title node
  │     │  └─ _BulletPoints[-]                            ← bullet node
  │     │     ├─ "Output MUST be valid JSON."             ← leaf
  │     │     ├─ "Do NOT include any text outside JSON."  ← leaf
  │     │     ├─ "Do NOT wrap output in markdown."        ← leaf
  │     │     ├─ "Always follow the schema exactly."      ← leaf
  │     │     ├─ "If a field has no data, return…"        ← leaf
  │     │     ├─ "Relevance scores must be between 0…"    ← leaf
  │     │     └─ "Topics should reflect main themes…"     ← leaf
  │     ├─ _Section[MULTILINGUAL SUPPORT]                 ← title node
  │     │  ├─ "Input text can be in ANY language."        ← leaf
  │     │  ├─ "All your analysis must be in ENGLISH."     ← leaf
  │     │  └─ _Array
  │     │     ├─ "The output fields must follow…"         ← leaf
  │     │     └─ _BulletPoints[-]                         ← bullet node
  │     │        ├─ '"topics.name": must be in ENGLISH'   ← leaf
  │     │        ├─ '"key_phrases": same lang as input'   ← leaf
  │     │        ├─ '"summary": must be in ENGLISH'       ← leaf
  │     │        └─ '"client_sentiment": never translated'← leaf
  │     ├─ _Section[SENTIMENT RULES]                      ← title node
  │     │  └─ _BulletPoints[-]                            ← bullet node
  │     │     ├─ "positive: satisfaction, praise…"        ← leaf
  │     │     ├─ "negative: complaints, anger…"           ← leaf
  │     │     ├─ "neutral: factual, balanced…"            ← leaf
  │     │     └─ "mixed: conflicting emotions…"           ← leaf
  │     ├─ _Section[TOPIC RULES]                          ← title node
  │     │  └─ _BulletPoints[-]                            ← bullet node
  │     │     ├─ "Extract 3-7 topics max"                 ← leaf
  │     │     ├─ "Avoid duplicates or similar topics"     ← leaf
  │     │     ├─ "Use concise noun phrases"               ← leaf
  │     │     └─ "Score by importance in the text"        ← leaf
  │     ├─ _Section[KEY PHRASES RULES]                    ← title node
  │     │  └─ _BulletPoints[-]                            ← bullet node
  │     │     ├─ "Extract 3-10 important phrases"         ← leaf
  │     │     ├─ "Must be near-exact from input text"     ← leaf
  │     │     └─ "Avoid repetition of topics"             ← leaf
  │     └─ _Section[SUMMARY RULES]                        ← title node
  │        └─ _BulletPoints[-]                            ← bullet node
  │           ├─ "Write a short, clear summary."          ← leaf
  │           └─ "Capture main intent and outcome…"       ← leaf
```

Here, the prompt is parsed into distinct components:

- **Sections** for markdown headers (which can easily be swapped for XML tags).
    
- **Arrays** for sequences of related elements.
    
- **BulletPoints** for list items.
    
- **Fields** for expected output structure definitions.
    
- **Identities** for the leaf-node sentences themselves.
    

**Pruning the Tree: Prompt Ablation Studies**

If we accept this tree view, a natural question arises: **What happens to my performance if I remove a specific node?** Concretely speaking, what happens if I remove the leaf _"Avoid repetition of topics"_? Or the entire _"KEY PHRASES RULES"_ section? Does performance decrease, stay the same, or actually improve?

To answer this, we can perform an **ablation study** on all the nodes of our prompt tree. To do this, we need a way to measure the error rate over a test dataset. This will highly depend on your business needs. For this case, I'll use two metrics:

1. **Structure:** The output must be valid JSON parseable by my schema.
    
2. **Language:** The output language must strictly be English (or the language provided).
    

We can compute the value of any specific line of text by measuring the performance of the full prompt, and then subtracting the performance of the prompt _without_ that specific node:

$Contribution = Metric_{baseline} - Metric_{AblatedNode}$

Let's look at the actual terminal output of such a test:

Plaintext

```
============================================================
  language_match[english]
============================================================
ChatMessages
├─ ChatMessage[system]
│  └─ _Array                                                                  0.000  ░░░░░░░░░░░░
│     └─ _Section[Instructions]                                               0.000  ░░░░░░░░░░░░
│        └─ _Array                                                            0.000  ░░░░░░░░░░░░
│           ├─ _Array                                                         0.167  ██░░░░░░░░░░
│           │  ├─ "You are a text analysis engine.↵↵"                         0.000  ░░░░░░░░░░░░
│           │  ├─ "Your task is to analyze the provided text an…"             0.000  ░░░░░░░░░░░░
│           │  ├─ "that strictly follows the required schema.↵↵"              0.200  ██░░░░░░░░░░
│           │  ├─ "You must ALWAYS return valid JSON only. "                  0.167  ██░░░░░░░░░░
│           │  └─ "Do not include explanations, markdown, or ex…"             0.333  ████░░░░░░░░
...
├─ ChatMessage[user]
│  └─ _Array                                                                  1.000  ████████████
│     └─ _WrappedContent[triple]                                              0.000  ░░░░░░░░░░░░
│        └─ "Plût au ciel que le lecteur, enhardi et deve…"                   1.000  ████████████
└─ ChatMessage[assistant]
   └─ _Array                                                                  0.000  ░░░░░░░░░░░░
      └─ "Below is the output json with fields in ENGL…"                      0.000  ░░░░░░░░░░░░
```

**Understanding the Results**

The score on the right represents the contribution of that specific component.

Notice the `ChatMessage[user]` node at the bottom scores a `1.000` (100% impact). This makes sense—if we ablate the user's input text, the entire task fundamentally fails.

However, look at the very last line: `ChatMessage[assistant]: "Below is the output json with fields in ENGLISH:"`. The score is `0.000`. This gives us insights that pre-filling the assistant's response with this phrasing provides **absolutely no value** for language stability. Therefore, we have empirical rationale to justify dropping it, saving tokens and latency! 

This programmatic approach allows for true, data-driven **prompt profiling**, letting developers strip away the "voodoo" of prompt engineering and better understands elements to keeps, and elements to remove.

### **Conclusion**
 Prompt engineering doesn't have to be a guessing game based on "vibes". By treating prompts as Abstract Syntax Trees and running automated ablation studies, it allows us to properly profile our prompt, as we do profile any of our program in software development.

 While this example focuses on structural validity and language matching, this framework can easily be expanded for any metrics, deterministic or AI based.

I've implemented a short POC repo [here](https://github.com/BaptGau/prompt-profiling). Note that this is still a POC. What would allow it to move to a proper feature would be, in my opinion:

- Setting up a parser, to convert any prompt in an abstract syntax tree (as it can be a bit tricky to write manually)
- Having something that works and aggregates metrics on a dataset level (current only work on one sample).
- Having a proper statistical significance measure. As LLM outputs are random variables, it means all metrics outputs are as well. Some of them, like latency or tokens are highly volatile - we'll never see 2 same results in 2 different runs - therefore a need of statistical robustness metrics implementation is needed to use it as a proper feature.