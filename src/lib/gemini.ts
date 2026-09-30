/**
 * Google Gemini API Client
 * Supports Gemini 3.8 Flash (or configured model) for Multimodal Vision & Text Q&A
 * along with Google AI Studio Text Embeddings (text-embedding-004)
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

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${apiKey}`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "models/text-embedding-004",
      content: {
        parts: [{ text: text.slice(0, 8000) }]
      }
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini Embedding API Error (${response.status}): ${errText}`);
  }

  const data = await response.json();
  return data.embedding?.values || [];
}

export async function generateGeminiDiagnosis(payload: {
  userQuery: string;
  contextChunks: string;
  imageBase64?: string;
  imageMimeType?: string;
}): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";

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

  // If API key is not yet set, provide a helpful mock response grounded in the knowledge base
  if (!apiKey) {
    return `### 🌿 PlantDoc Clinical Diagnosis (Demo Mode)

> **Note:** \`GEMINI_API_KEY\` is not yet set in \`.env.local\`. Showing grounded analysis from the built-in textbook knowledge base:

${payload.contextChunks ? `**Retrieved Pathology Knowledge:**\n${payload.contextChunks.slice(0, 600)}...\n` : ''}

**Setup Instructions:**
To unlock full live Gemini 3.8 Flash Multimodal analysis and streaming, get your free key at [Google AI Studio](https://aistudio.google.com/) and paste it into \`.env.local\`.`;
  }

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

  if (!response.ok) {
    const errText = await response.text();
    // Fallback gracefully if model name is different in certain regions
    if (response.status === 404 && model !== "gemini-1.5-flash") {
      const fallbackEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
      const fbRes = await fetch(fallbackEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents,
          systemInstruction: { parts: [{ text: systemPrompt }] },
          generationConfig: { temperature: 0.3, maxOutputTokens: 2048 }
        })
      });
      if (fbRes.ok) {
        const fbData = await fbRes.json();
        return fbData.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";
      }
    }
    throw new Error(`Gemini API Error (${response.status}): ${errText}`);
  }

  const data = await response.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated from PlantDoc.";
}
