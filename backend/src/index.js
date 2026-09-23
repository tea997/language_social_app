// ============================================
// index.js - PDF ko load karke Pinecone mein store karna
// Yeh file PDF document ko read karti hai, chunks mein todti hai,
// embeddings banati hai, aur Pinecone vector database mein store karti hai
// ============================================

// Step 1: Environment variables load karo (.env file se)
// .env file mein API keys aur config hota hai
import * as dotenv from 'dotenv';
dotenv.config();

// Step 2: Required libraries import karo
// PDFLoader - PDF file ko read karne ke liye
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// RecursiveCharacterTextSplitter - Bade text ko chhote chunks mein todne ke liye
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
// GoogleGenerativeAIEmbeddings - Text ko vector (numbers) mein convert karne ke liye
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
// Pinecone - Vector database client
import { Pinecone } from '@pinecone-database/pinecone';
// PineconeStore - LangChain ka Pinecone integration (embedding + store ek saath)
import { PineconeStore } from '@langchain/pinecone';

// ============================================
// Main function - Poora indexing process yahan hota hai
// ============================================
async function indexDocument() {
  // ------------------------------------------
  // STEP 1: PDF Load karo
  // ------------------------------------------
  const PDF_PATH = './rag_test.pdf'; // PDF file ka path
  const pdfLoader = new PDFLoader(PDF_PATH); // PDFLoader instance banao
  const rawDocs = await pdfLoader.load(); // PDF ko read karo, pages ka array milega
  console.log(`✅ PDF loaded (${rawDocs.length} pages)`);

  // ------------------------------------------
  // STEP 2: Chunking - Bade text ko chhote pieces mein todo
  // ------------------------------------------
  // Kyun? LLM ek baar mein bahut bada text process nahi kar sakta
  // Isliye hum 1000 characters ke chunks banate hain
  // 200 character overlap rakhte hain taaki context na tute
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,    // Har chunk max 1000 characters ka hoga
    chunkOverlap: 200,  // Har chunk ke beech 200 characters common honge
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  console.log(`✅ Chunking completed (${chunkedDocs.length} chunks)`);

  // ------------------------------------------
  // STEP 3: Embedding Model Configure karo
  // ------------------------------------------
  // Embedding = Text ko numbers (vector) mein convert karna
  // "Hello" → [0.12, -0.45, 0.78, ...] (3072 numbers)
  // Similar meaning wale texts ke vectors bhi similar hote hain
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,  // Google API key (.env se)
    model: 'gemini-embedding-001',       // Google ka embedding model
  });

  // ------------------------------------------
  // STEP 3.1: Rate Limit Handling
  // ------------------------------------------
  // Problem: Free tier mein sirf 100 embedding requests/minute allowed hain
  // Default LangChain code saare batches ek saath bhejta hai → rate limit error
  // Solution: Ek-ek batch sequentially bhejo with delay aur retry logic
  embeddings._embedDocumentsContent = async function (documents) {
    const BATCH_SIZE = 5;         // Ek batch mein 5 documents embed karo
    const DELAY_MS = 3000;        // Har batch ke baad 3 second ruko
    const MAX_RETRIES = 3;        // Rate limit pe max 3 baar retry karo
    const allEmbeddings = [];     // Saare embeddings yahan collect honge
    const totalBatches = Math.ceil(documents.length / BATCH_SIZE);

    for (let i = 0; i < documents.length; i += BATCH_SIZE) {
      const batch = documents.slice(i, i + BATCH_SIZE);
      const batchNum = Math.floor(i / BATCH_SIZE) + 1;

      // Har document ko Google API ke format mein convert karo
      const requests = batch.map((doc) => this._convertToContent(doc));

      // Retry loop - agar rate limit aaye toh wait karke dobara try karo
      let success = false;
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
          // Google API ko batch embedding request bhejo
          const res = await this.client.batchEmbedContents({ requests });
          // Response se vector values nikalo aur array mein daalo
          allEmbeddings.push(...res.embeddings.map((e) => e.values || []));
          console.log(`  📦 Embedded batch ${batchNum}/${totalBatches}`);
          success = true;
          break; // Success → retry loop se bahar aao
        } catch (err) {
          // Agar 429 (Too Many Requests) error aaya → wait karo aur retry karo
          if (err.message.includes('429') || err.message.includes('quota')) {
            const waitTime = 15 + attempt * 10; // 15s, 25s, 35s
            console.log(`  ⏳ Rate limited on batch ${batchNum}, waiting ${waitTime}s (attempt ${attempt + 1}/${MAX_RETRIES})...`);
            await new Promise((r) => setTimeout(r, waitTime * 1000));
          } else {
            // Koi aur error → retry mat karo, fail karo
            console.error(`  ❌ Batch ${batchNum} failed:`, err.message);
            break;
          }
        }
      }

      // Agar 3 retries ke baad bhi fail → poora process rok do
      if (!success) {
        throw new Error(`Embedding failed at batch ${batchNum} after ${MAX_RETRIES} retries`);
      }

      // Har batch ke baad 3 second ruko (rate limit se bachne ke liye)
      await new Promise((r) => setTimeout(r, DELAY_MS));
    }

    return allEmbeddings;
  }.bind(embeddings);

  console.log('✅ Embedding model configured (gemini-embedding-001, 3072 dimensions)');

  // ------------------------------------------
  // STEP 4: Pinecone Vector Database Configure karo
  // ------------------------------------------
  // Pinecone = Cloud vector database jahan embeddings store hote hain
  // Baad mein in vectors ko search karke similar documents dhoond sakte hain
  const pinecone = new Pinecone(); // Auto-reads PINECONE_API_KEY from .env
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME); // Index select karo
  console.log(`✅ Pinecone configured (index: ${process.env.PINECONE_INDEX_NAME})`);

  // ------------------------------------------
  // STEP 5: Embed + Store - Chunks ko embed karke Pinecone mein daalo
  // ------------------------------------------
  // PineconeStore.fromDocuments yeh 3 kaam ek saath karta hai:
  //   1. Har chunk ka text → embedding (vector) mein convert
  //   2. Pinecone mein upsert (store/update)
  //   3. maxConcurrency = 2 → Pinecone ko 2 requests ek saath bhejo
  try {
    console.log('🔄 Embedding and storing documents...');
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 2, // Pinecone ko ek baar mein 2 requests bhejo
    });
    console.log('✅ All data stored successfully in Pinecone! 🎉');
  } catch (error) {
    console.error('❌ Error storing data:', error.message);
  }
}

// Main function ko call karo
indexDocument();
 