from engine.pyalice import *
import numpy as np
import cv2

class OpencvRccorder(Codelet):
    def start(self):
        self.rx_color = self.isaac_proto_rx("ColorCameraProto", "color")
        self.color_writer = cv2.VideoWriter(
            '/home/misha/Desktop/output.mp4', cv2.VideoWriter_fourcc(*"MP4V"), 
            25.0, (1280,720)
        )
        # This part will be run once in the beginning of the program
        # We can tick periodically, on every message, or blocking. See documentation for details.
        self.tick_on_message(self.rx_color)

    def stop(self):
        self.log_info("Releasing writers..")
        self.color_writer.release()

    def tick(self):
        # This part will be run at every tick. We are ticking periodically in this example.

        # Print out message to console. The message is set in ping_python.app.json file.
        self.log_info(self.get_isaac_param("message"))
    
        if not self.rx_color.available():
            return
        color_proto = self.rx_color.get_proto()
        color_image = cv2.cvtColor(np.frombuffer(
                self.rx_color.get_buffer_content(
                    color_proto.image.dataBufferIndex
                ),
                np.uint8
            ).reshape(
                color_proto.image.rows, 
                color_proto.image.cols, 
                color_proto.image.channels
            ), 
            cv2.COLOR_BGR2RGB
        )

        #img_np = cv2.cvtColor(
        #    image_data, cv2.COLOR_
        #) # cv2.IMREAD_COLOR in OpenCV 3.1
        self.log_info("Got img: %s %s %s" % (color_proto, 'img_np', color_image.shape))
        #cv2.imwrite('/tmp/image.png',color_image)
        self.color_writer.write(color_image)
        #   auto input = rx_input_image().getProto();
        
        # ImageConstView3ub input_image;
        # bool ok = FromProto(input.getImage(), rx_input_image().buffers(), input_image);
        # ASSERT(ok, "Failed to deserialize the input image");

        # const size_t rows = input_image.rows();
        # const size_t cols = input_image.cols();
        # Image1ub output_image(rows, cols);
        # cv::Mat image =
        #     cv::Mat(rows, cols, CV_8UC3,
        #             const_cast<void*>(static_cast<const void*>(input_image.data().pointer())));

    


def main():
    app = Application(app_filename="packages/opencv_recorder/opencv_recorder.app.json")
    app.register({"opencv_recorder_node": OpencvRccorder})
    app.start_wait_stop()


if __name__ == '__main__':
    main()
