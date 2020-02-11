import express from 'express';
import bodyParser from 'body-parser';

const app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const router = express.Router();

app.set('port', (process.env.PORT || 3000));

app.use(function (req, res, next) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});


router.get('', (req, res) => {
  res.status(200).send('Hello world');
});

app.use(router);

app.listen(app.get('port'), function () {
  console.log('Node app is running at localhost:' + app.get('port'));
});
