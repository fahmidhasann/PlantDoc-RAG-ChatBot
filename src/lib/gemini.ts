/**
 * Google Gemini API Client
 * Supports Gemini 3.8 Flash (with automatic fallback to Gemini 3.5 Flash)
 * and Google AI Studio Text Embeddings (gemini-embedding-001 with 768 dims)
 */

export interface MultimodalPayload {
  prompt: string;
  contextText?: string;
  imageBufferBase64?: string;
  imageMimeType?: string;
}

export async function generateGeminiEmbedding(text: string): Promise<number[]> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${apiKey}`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "models/gemini-embedding-001",
        content: {
          parts: [{ text: text.slice(0, 8000) }]
        },
        outputDimensionality: 768
      })
    });

    if (response.ok) {
      const data = await response.json();
      return data.embedding?.values || [];
    }
  } catch (err) {
    console.warn("Embedding generation warning:", err);
  }

  return [];
}

export async function generateGeminiDiagnosis(payload: {
  userQuery: string;
  contextChunks: string;
  imageBase64?: string;
  imageMimeType?: string;
}): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;

  const systemPrompt = `You are PlantDoc, an elite plant pathologist AI expert trained on Agrios' Plant Pathology textbook (5th Edition).
Your mission is to provide accurate, clinical, and actionable diagnosis for plant diseases based on the provided textbook context.

Guidelines:
1. Ground your answer in the provided textbook context.
2. When answering, cite exact page numbers (e.g., "[Agrios Plant Pathology, Page 421]").
3. If an image is provided:
   - Carefully inspect visual leaf symptoms (lesions, chlorosis, halo, sporulation, vein necrosis).
   - Correlate visual findings with the retrieved textbook disease descriptions.
4. Structure your response clearly:
   - 🔍 **Diagnosis & Causal Organism**: Name of disease & pathogen (scientific binomial).
   - 🌿 **Observed Symptoms**: What indicates this disease.
   - 🛡️ **Actionable Management**:
     - *Cultural / Sanitation*
     - *Organic / Biological Controls*
     - *Chemical Treatments (Active ingredients & fungicides)*
   - 📖 **Textbook Reference**: Page number citations.
5. If the context does not contain sufficient details, state what is known and advise consulting a local agricultural extension officer.`;

  const contents: any[] = [];
  const parts: any[] = [];

  // Add textbook context
  parts.push({
    text: `### Verified Textbook Context Chunks:\n${payload.contextChunks}\n\n---\n`
  });

  // If image is provided, attach inline data
  if (payload.imageBase64 && payload.imageMimeType) {
    parts.push({
      inline_data: {
        mime_type: payload.imageMimeType,
        data: payload.imageBase64
      }
    });
    parts.push({
      text: `User uploaded a photo of an affected plant leaf. Please examine this image and answer the user's query: "${payload.userQuery || 'Diagnose the disease in this plant image and suggest treatment.'}"`
    });
  } else {
    parts.push({
      text: `User Question: "${payload.userQuery}"`
    });
  }

  contents.push({
    role: "user",
    parts: parts
  });

  let lastErrorMessage = "";

  if (apiKey) {
    const modelsToTry = [
      process.env.GEMINI_MODEL,
      "gemini-3.8-flash",
      "gemini-3.5-flash"
    ].filter(Boolean) as string[];

    for (const model of modelsToTry) {
      try {
        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents,
            systemInstruction: {
              parts: [{ text: systemPrompt }]
            },
            generationConfig: {
              temperature: 0.3,
              maxOutputTokens: 2048
            }
          })
        });

        if (response.ok) {
          const data = await response.json();
          return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated from PlantDoc.";
        }

        const errText = await response.text();
        console.warn(`Model ${model} returned ${response.status}:`, errText);

        if (errText.includes("leaked")) {
          lastErrorMessage = "Google AI Studio has deactivated this API key because it was reported in a public leak. Please generate a fresh key at aistudio.google.com.";
          break;
        } else if (response.status === 429) {
          lastErrorMessage = "Gemini API rate limit reached (15 requests/min on Free Tier). Please retry in 30 seconds.";
        } else {
          lastErrorMessage = `Gemini API Error (${response.status}): ${errText}`;
        }
      } catch (err: any) {
        console.warn(`Model ${model} call failed:`, err);
        lastErrorMessage = err.message || "Network error contacting Gemini API.";
      }
    }
  } else {
    lastErrorMessage = "GEMINI_API_KEY is not configured.";
  }

  // Graceful Fallback: Generate structured diagnosis directly from the retrieved textbook context
  // This guarantees the app NEVER crashes and always presents clinical answers to users!
  const alertBanner = lastErrorMessage.includes("leaked")
    ? `> ⚠️ **API Key Notice**: Google AI Studio has automatically deactivated the previous API key because it was detected in a public leak. A fresh key can be generated at [Google AI Studio](https://aistudio.google.com/app/apikey). In the meantime, PlantDoc is displaying direct clinical analysis from the 948-page textbook index:`
    : lastErrorMessage.includes("rate limit")
    ? `> ⏳ **Rate Limit Notice**: Free tier rate limit reached. Displaying grounded textbook knowledge:`
    : `> ℹ️ **Textbook Grounding Mode**: Grounded directly in Agrios' Plant Pathology textbook database:`;

  return `### 🔍 Grounded Clinical Pathology Summary

${alertBanner}

${payload.contextChunks ? `${payload.contextChunks}` : 'No matching disease profile found for this specific query.'}

---

### 🛡️ General Management Principles
* **Sanitation**: Remove and destroy infected plant debris to eliminate inoculum reservoirs.
* **Crop Rotation**: Implement 2–3 year non-host crop rotations.
* **Chemical Strategy**: Alternate contact protectants (e.g. Mancozeb, Copper) with systemic targeted fungicides to prevent resistance.`;
}
