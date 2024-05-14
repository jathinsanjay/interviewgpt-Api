const express=  require("express")
const app = express()
const bodyParser = require('body-parser');
const port = 8000
const cors= require('cors')
app.use(cors())

app.use(bodyParser.json({ limit: '50mb' }));

app.use(express.json())
const API_KEY= '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'
app.post('/questions/:topic/:level', async (req, res) => {
  const { topic, level } = req.params;
app.get("/",(req,res)=>{
  res.send("hello")
})

  const options = {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages: [{
        role: 'user',
        content: `
          generate 23 ${level} MULTIPLE CHOICE interview questions on ${topic} using the following format, use code variable only if available:
         answer must be justion the option alphabet
          Q: [give question here] ? 
          [give code if exists else don't create a term code]
          Options:
          A)[Answer A] 
          B) [Answer B] 
          C)[Answer C]
          D) [Answer D]
          Answer:[option]
        `
      }],
      max_tokens:1500,
    },)
  };

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', options);
    const data = await response.json();
    res.send(data);
  } catch (error) {
    console.log(error);
    res.status(500).send('Error fetching questions');
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
