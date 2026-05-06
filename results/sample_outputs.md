# RAG Evaluation Sample Outputs

## Report Notes

### RAG Chain Implementation
The RAG chain connects the semantic retriever with the language model to generate document-grounded answers. For each user query, the retriever returns the top-k most relevant document chunks from the vector store. These chunks are inserted into a structured prompt that instructs the language model to answer only using the provided context. If the retrieved documents do not contain enough information, the model is instructed to state that the information is unavailable. This design reduces hallucination and improves the traceability of generated answers.

### Evaluation Design
The evaluation was designed to assess both retrieval quality and answer generation quality. A set of test questions was created to cover direct factual queries, comparison-based queries, recommendation-style queries, and out-of-context queries. The current prototype uses a 20,000-row sampled product dataset and retrieval examples from the project notebook, including queries about waterproof eyebrow products, powder item form, Cherioll brand products, long lasting makeup, and highly rated beauty products. For each question, the system records the generated answer, retrieved context, and evaluation scores. The outputs are saved in a CSV file for structured analysis and in a Markdown file for qualitative inspection.

### Evaluation Metrics
Three evaluation criteria were used: relevance, faithfulness, and completeness. Relevance measures whether the answer directly addresses the user question. Faithfulness measures whether the answer is supported by the retrieved context and avoids unsupported claims. Completeness measures whether the answer provides sufficient detail based on the available evidence. These criteria are scored on a five-point scale, where a higher score indicates better RAG performance.

---

## Question
What waterproof eyebrow products have a natural finish?

## Answer
Based on the retrieved documents, the relevant products are:
1. Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (AUBURN) (rating 3.9; price N/A; store EINHOOM; source 12763).
2. Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (BRUNETTE) (rating 4.0; price N/A; store EINHOOM; source 13508). 1 retrieved product(s) were not included because the retrieved fields did not explicitly match all requested attributes.

## Retrieved Evidence
- Source 12763; title: Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (AUBURN); rating: 3.9; price: N/A; store: EINHOOM; similarity: 0.6959
- Source 13508; title: Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (BRUNETTE); rating: 4.0; price: N/A; store: EINHOOM; similarity: 0.6920
- Source 16366; title: ANTIKE Makeup Eyebrow Stamp Kit Waterproof Brow Stamp and Eyebrow Stencil Kit Makeup Tools Eyebrow Powder with 12 Reusable Eyebrow Stencils (Dark Brown) LB000251-2; rating: 3.5; price: N/A; store: ANTIKE; similarity: 0.6797

## Scores
- Relevance: 5
- Faithfulness: 3
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which beauty products have powder as the item form?

## Answer
Based on the retrieved documents, the relevant products are:
1. Laura Mercier Secret Brightening Powder for Under Eyes, Color :Shade 1 -fair to medium skin (rating 4.4; price N/A; store B.B Beauty; source 14648).
2. Face Powder Blush Cruelty Free – Seasons Blush Powder for Cheeks with Shimmer Matte Finish – 0.12 Oz Natural, Vegan, Organic, Blendable, Idea Gift for Girlfriend, Sister, Wife and Daily Use (PEACH) (rating 4.2; price N/A; store N/W; source 11749).
3. [I'm from] Rice Beauty Mask 3.88 Oz, Renewed, Gentle Exfoliating Scrub (rating 4.3; price 23.77; store I'm from; source 16417).

## Retrieved Evidence
- Source 14648; title: Laura Mercier Secret Brightening Powder for Under Eyes, Color :Shade 1 -fair to medium skin; rating: 4.4; price: N/A; store: B.B Beauty; similarity: 0.6729
- Source 11749; title: Face Powder Blush Cruelty Free – Seasons Blush Powder for Cheeks with Shimmer Matte Finish – 0.12 Oz Natural, Vegan, Organic, Blendable, Idea Gift for Girlfriend, Sister, Wife and Daily Use (PEACH); rating: 4.2; price: N/A; store: N/W; similarity: 0.6488
- Source 16417; title: [I'm from] Rice Beauty Mask 3.88 Oz, Renewed, Gentle Exfoliating Scrub; rating: 4.3; price: 23.77; store: I'm from; similarity: 0.6300

## Scores
- Relevance: 5
- Faithfulness: 5
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
What products are from the Cherioll brand?

## Answer
Based on the retrieved documents, the relevant products are:
1. Tattoo Eyebrow Stickers, Waterproof Eyebrow, 4D Imitation Eyebrow Tattoos, 4D Hair-like Authentic Eyebrows Waterproof Long Lasting for Woman & Man Makeup Tool (rating 3.1; price N/A; store Cherioll; source 0).
2. Nail Repair size4… (rating 4.3; price N/A; store Cherioll; source 1669).

## Retrieved Evidence
- Source 5019; title: The Beauty Crop Oui Cherie Lip Oil - Hydrates your Lips with Cherry Lip Oil (Sacrebleu); rating: 5.0; price: N/A; store: Generic; similarity: 0.5859
- Source 0; title: Tattoo Eyebrow Stickers, Waterproof Eyebrow, 4D Imitation Eyebrow Tattoos, 4D Hair-like Authentic Eyebrows Waterproof Long Lasting for Woman & Man Makeup Tool; rating: 3.1; price: N/A; store: Cherioll; similarity: 0.5450
- Source 1669; title: Nail Repair size4…; rating: 4.3; price: N/A; store: Cherioll; similarity: 0.5424

## Scores
- Relevance: 5
- Faithfulness: 4
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which makeup products are described as long lasting?

## Answer
Based on the retrieved documents, the relevant products are:
1. Makeup Depot Waterproof Long lasting Smudgeproof Makeup Tattoo Liquid Eyeliner 24 hr Lasting Eyeliner - Matte Black (rating 3.7; price 4.99; store MAKEUP DEPOT; source 19521).
2. Static 16 Hour Long Lasting Dual Ended Lip Color Lipstick - daniella (rating 2.4; price 24.0; store DRQ; source 3763).

## Retrieved Evidence
- Source 19521; title: Makeup Depot Waterproof Long lasting Smudgeproof Makeup Tattoo Liquid Eyeliner 24 hr Lasting Eyeliner - Matte Black; rating: 3.7; price: 4.99; store: MAKEUP DEPOT; similarity: 0.5925
- Source 3763; title: Static 16 Hour Long Lasting Dual Ended Lip Color Lipstick - daniella; rating: 2.4; price: 24.0; store: DRQ; similarity: 0.5401
- Source 14051; title: Linatural™ Broad Spectrum Preservative Ingredient for DIY Vitamin C, Ascorbic Acid & Hyaluronic Acid Powder Serum, Facial Lotion & Other Cosmetic Making Formulations, 1 oz; rating: 4.6; price: N/A; store: RESURRECTIONbeauty; similarity: 0.5396

## Scores
- Relevance: 5
- Faithfulness: 4
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which all beauty products have high ratings?

## Answer
Based on the retrieved documents, the relevant products are:
1. Beauty Creations Ultra Dazzle Lipgloss Golden Girl (rating 4.8; price 8.99; store Beauty Creations; source 402).
2. Laura Mercier Secret Brightening Powder for Under Eyes, Color :Shade 1 -fair to medium skin (rating 4.4; price N/A; store B.B Beauty; source 14648). Products below a 4.0 average rating were excluded.

## Retrieved Evidence
- Source 402; title: Beauty Creations Ultra Dazzle Lipgloss Golden Girl; rating: 4.8; price: 8.99; store: Beauty Creations; similarity: 0.6648
- Source 14648; title: Laura Mercier Secret Brightening Powder for Under Eyes, Color :Shade 1 -fair to medium skin; rating: 4.4; price: N/A; store: B.B Beauty; similarity: 0.6619
- Source 3763; title: Static 16 Hour Long Lasting Dual Ended Lip Color Lipstick - daniella; rating: 2.4; price: 24.0; store: DRQ; similarity: 0.6407

## Scores
- Relevance: 4
- Faithfulness: 4
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which waterproof eyebrow product would you recommend based on the retrieved ratings?

## Answer
Based on the retrieved evidence, I would select Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (BRUNETTE) (rating 4.0; price N/A; store EINHOOM; source 13508).

## Retrieved Evidence
- Source 12763; title: Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (AUBURN); rating: 3.9; price: N/A; store: EINHOOM; similarity: 0.7031
- Source 13508; title: Semi-permanent Eyebrow Gel for Waterproof, Long Lasting, Smudge-Proof, Sweat Resistant, Full Natural, Brow Pen Tinted Makeup (BRUNETTE); rating: 4.0; price: N/A; store: EINHOOM; similarity: 0.6989
- Source 17759; title: Tattoo Eyebrows,Fake Eyebrows,Eyebrow Tattoo Sticker,Waterproof Eyebrow Tattoos Stickers,Suitable Sizes,Waterproof and Long-Lasting; rating: 1.7; price: N/A; store: LDREAMAM; similarity: 0.6794

## Scores
- Relevance: 5
- Faithfulness: 4
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which products are suitable for customers looking for long lasting makeup?

## Answer
Based on the retrieved documents, the relevant products are:
1. Makeup Depot Waterproof Long lasting Smudgeproof Makeup Tattoo Liquid Eyeliner 24 hr Lasting Eyeliner - Matte Black (rating 3.7; price 4.99; store MAKEUP DEPOT; source 19521).

## Retrieved Evidence
- Source 19521; title: Makeup Depot Waterproof Long lasting Smudgeproof Makeup Tattoo Liquid Eyeliner 24 hr Lasting Eyeliner - Matte Black; rating: 3.7; price: 4.99; store: MAKEUP DEPOT; similarity: 0.5829
- Source 11450; title: Makeup Brush Mini 4 in 1 Kit for Travel Cosmetic Applicator Foundation Sponges Brushes Valentine's Day Gift, Blending Eyeshadow Concealer Eyeliner Eyebrow; rating: 3.7; price: N/A; store: Seencool; similarity: 0.5610
- Source 3606; title: Almay Clear Complexion Makeup, Hypoallergenic, Cruelty Free, Fragrance Free, Dermatologist Tested Foundation, 1oz; rating: 4.4; price: 13.99; store: Almay; similarity: 0.5566

## Scores
- Relevance: 5
- Faithfulness: 4
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
Which retrieved product has the best balance between price and rating?

## Answer
Only one retrieved product includes both price and rating, so the system cannot reliably compare the best balance. The available product is Byredo Mojave Ghost by Byredo Vial (sample) .06 oz (rating 4.2; price 23.3; store Byredo; source 15357).

## Retrieved Evidence
- Source 4180; title: Icy Hot Balm Size 3.5z Icy Hot Extra Strength Balm; rating: 4.1; price: N/A; store: Icy Hot; similarity: 0.4786
- Source 15357; title: Byredo Mojave Ghost by Byredo Vial (sample) .06 oz; rating: 4.2; price: 23.3; store: Byredo; similarity: 0.4767
- Source 10336; title: e.l.f. Lock On Lip Primer 0.1 fl oz, pack of 1; rating: 4.6; price: N/A; store: e.l.f.; similarity: 0.4745

## Scores
- Relevance: 5
- Faithfulness: 3
- Completeness: 5
- Notes: Rule-based scores based on question-answer overlap, answer-evidence overlap, answer formatting, and refusal behavior.

---

## Question
What will the product price be next year?

## Answer
The provided documents do not contain enough information to answer this question.

## Retrieved Evidence
- Source 19769; title: Morgan Taylor Forever Fabulous Marilyn Monroe Holiday/Winter 2018 Collection (Glam Mini 4 Pack); rating: 4.2; price: N/A; store: None; similarity: 0.3516
- Source 19130; title: Skin Ceuticals Sport Uv Defense Spf 50, 3 Ounce; rating: 4.8; price: N/A; store: SKINCEUTICALS; similarity: 0.3479
- Source 17850; title: 1000 Hour Hair Color Stick Instant Touch-Up For Grey Hair 4 Colors! (Light Brown); rating: 5.0; price: N/A; store: 1000Hour; similarity: 0.3473

## Scores
- Relevance: 5
- Faithfulness: 5
- Completeness: 5
- Notes: Correct refusal for an out-of-context question.

---

## Question
What is the manufacturer's future business strategy?

## Answer
The provided documents do not contain enough information to answer this question.

## Retrieved Evidence
- Source 7025; title: Shaving Factory Blade Dispenser; rating: 4.6; price: 477.06; store: Shaving Factory; similarity: 0.2838
- Source 19848; title: QTMY 2022 Baroque Pearls Flower Headband for Women,Wide Hair Hoop Hair Accessories Head Band Headwear; rating: 5.0; price: N/A; store: QTMY; similarity: 0.2787
- Source 4372; title: Replacement Blade Compatible with Philips Oneblade Series - Pack of 3; rating: 2.6; price: N/A; store: RABLATEC; similarity: 0.2770

## Scores
- Relevance: 5
- Faithfulness: 5
- Completeness: 5
- Notes: Correct refusal for an out-of-context question.

---

## Question
Does the dataset contain warranty information?

## Answer
The provided documents do not contain enough information to answer this question.

## Retrieved Evidence
- Source 2960; title: Spectral DNC-N by DS Laboratories Hair Loss and Thinning Hair Product - DHT Blocker, Anti Hair Loss Treatment, Hair Growth for Thicker, Fuller Hair, Water Based Formula; rating: 3.9; price: N/A; store: DS LABORATORIES; similarity: 0.2542
- Source 9557; title: Martiderm The Originals Proteos Hydra Plus Ampoules 30amp.; rating: 4.4; price: 49.91; store: Martiderm; similarity: 0.2416
- Source 1663; title: Martiderm Dsp-cover Spf50+ 4ml; rating: 3.7; price: 32.25; store: Martiderm; similarity: 0.2405

## Scores
- Relevance: 5
- Faithfulness: 5
- Completeness: 5
- Notes: Correct refusal for an out-of-context question.

---
