import { Router, Request, Response } from 'express';
import { RoutingError, Response as ResponseResult, Success } from '../model/exceptions.model';
import { Controller } from '../controller/controller';
export class Routes {
  router: Router;
  public constructor(router: Router) {
    this.router = router;
    this.addRoutes();
  }

  private addRoutes() {
    this.get('/', (req: Request) => {
      return Promise.resolve(new Success('Hello world!'));
    });

    this.get('/hello/:name', (req: Request) => {
      const controller = new Controller();
      return controller.hello(req.params.name);
    });

    this.router.use((req: Request, res: Response) => {
      res.send(new RoutingError('Route not found'));
    });
  }

  private get(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.get(uri, (req, res) => {
      result(req)
        .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
        .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
    });
  }

  private post(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.post(uri, (req, res) => {
      result(req)
        .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
        .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
    });
  }

  private patch(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.patch(uri, (req, res) => {
      result(req)
        .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
        .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
    });
  }

  private delete(uri: string, result: (request: Request) => Promise<ResponseResult>) {
    this.router.delete(uri, (req, res) => {
      result(req)
        .then((_result: { statusCode: number; }) => { res.status(_result.statusCode).send(_result); })
        .catch((error: { statusCode: any; }) => { res.status(error.statusCode || 500).send(error); });
    });
  }
}
