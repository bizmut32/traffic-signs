import { Router, Request, Response } from 'express';
import { RoutingError, Response as ResponseResult, Success, Error, ServerError } from '../model/exceptions.model';
import { Controller } from '../controller/controller';
export class Routes {
  router: Router;
  public constructor(router: Router) {
    this.router = router;

    this.addRoutes();

    this.router.use((req: Request, res: Response) => {
      res.send(new RoutingError('Route not found'));
    });
  }

  private addRoutes() {
    this.get('/', (req: Request) => {
      return Promise.resolve(new Success('Hello world!'));
    });

    this.get('/hello/:name', (req: Request) => {
      const controller = new Controller();
      return controller.hello(req.params.name);
    });

    this.post('/image', (req: Request) => {
      const controller = new Controller();
      this.require(req.body, 'image');
      return controller.classifyImage({image: req.body.image});
    });

    this.get('/image/random', (req: Request) => {
      const controller = new Controller();
      return controller.classifyRandomImage();
    });
  }

  private get(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.get(uri, (req, res) => {
      try {
        result(req)
          .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
          .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
      }
      catch (error) {
        res.status(error.statusCode || 500).send(error);
      }
    });
  }

  private post(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.post(uri, (req, res) => {
      try {
        result(req)
          .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
          .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
      }
      catch (error) {
        res.status(error.statusCode || 500).send(error);
      }    });
  }

  private patch(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.patch(uri, (req, res) => {
      try {
        result(req)
          .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
          .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
      }
      catch (error) {
        res.status(error.statusCode || 500).send(error);
      }    });
  }

  private delete(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.delete(uri, (req, res) => {
      try {
        result(req)
          .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
          .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
      }
      catch (error) {
        res.status(error.statusCode || 500).send(error);
      }    });
  }

  private require(body: any, properties: string) {
    const propertyArray = properties.split('.');
    let obj = body;
    for (const property of propertyArray) {
      if (obj[property] === undefined)
        throw new ServerError('Body does not have a parameter ' + properties);
      obj = obj[property];
    }
  }
}
