#!/bin/sh
mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-0
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/model_committee0.h5 \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-0

mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-1
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/model_committee1.h5 \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-1

mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-2
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/model_committee2.h5 \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-2

mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-3
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/model_committee3.h5 \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-3

mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-4
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/model_committee4.h5 \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/committee-4

mkdir /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/localizer
tensorflowjs_converter --input_format keras \
    /home/balassa/Documents/projects/traffic-signs/models/models/object-localizer \
    /home/balassa/Documents/projects/traffic-signs/traffic-signs-server/src/assets/models/localizer