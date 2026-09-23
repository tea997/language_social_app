import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import * as dotenv from 'dotenv';
import { processAndIndexPDF, queryRAG } from './services/ragService.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Enable CORS and JSON body parser
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps, curl, server-to-server)
      if (!origin) return callback(null, true);
      if (
        origin.startsWith('http://localhost') ||
        origin.endsWith('.vercel.app')
      ) {
        return callback(null, true);
      }
      return callback(null, true); // Fallback allow all origins for API flexibility
    },
    credentials: true,
  })
);
app.use(express.json({ limit: '50mb' }));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer storage config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf' || file.originalname.endsWith('.pdf')) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed!'), false);
    }
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

// Current active document session in memory
let activeDocument = {
  filename: 'rag_test.pdf',
  originalName: 'rag_test.pdf',
  pageCount: 112,
  chunkCount: 226,
  indexedAt: new Date().toISOString(),
};

// --- API ROUTES ---

// Health Check & Active Doc Status
app.get('/api/status', (req, res) => {
  res.json({
    status: 'online',
    pineconeIndex: process.env.PINECONE_INDEX_NAME,
    activeDocument,
  });
});

// Document Upload Endpoint
app.post('/api/upload', upload.single('document'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded.' });
    }

    const filePath = req.file.path;
    console.log(`\n📥 Received file upload: ${req.file.originalname}`);

    const result = await processAndIndexPDF(filePath, (statusMessage) => {
      console.log(`  🔄 Upload Status: ${statusMessage}`);
    });

    activeDocument = {
      filename: req.file.filename,
      originalName: req.file.originalname,
      pageCount: result.pageCount,
      chunkCount: result.chunkCount,
      indexedAt: new Date().toISOString(),
    };

    // Clean up uploaded temp file after indexing
    fs.unlink(filePath, (err) => {
      if (err) console.warn('Could not remove temp file:', err);
    });

    res.json({
      success: true,
      message: 'Document indexed successfully!',
      document: activeDocument,
    });
  } catch (error) {
    console.error('❌ Upload endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to process and index PDF.',
    });
  }
});

// Chat Query Endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { question, history } = req.body;
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Question string is required.' });
    }

    console.log(`\n💬 Received query: "${question}"`);
    const ragResult = await queryRAG(question, history || []);

    res.json({
      success: true,
      answer: ragResult.answer,
      standaloneQuery: ragResult.standaloneQuery,
      sourcesCount: ragResult.sourcesCount,
      activeDocument: activeDocument.originalName,
    });
  } catch (error) {
    console.error('❌ Chat endpoint error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Error executing query.',
    });
  }
});

app.listen(PORT, () => {
  console.log(`\n🚀 RAG Server listening on http://localhost:${PORT}`);
});
