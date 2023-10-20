const express=  require("express")
const app = express()
const port = 8000
const cors= require('cors')
app.use(cors())
app.use(express.json())
const API_KEY= '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'

app.post('/questions/:topic', async (req, res) => {
    const { topic } = req.params;
  
    const options = {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: `Give some interview mcq questions on ${topic} with options a,b,c,d on hard level` }],
        max_tokens: 300,
      }),
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