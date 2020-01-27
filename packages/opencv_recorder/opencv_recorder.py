from engine.pyalice import *
import numpy as np
import cv2
import click
from functools import partial


def handleTickException(func):
    def wrapped(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except Exception:
            self.logger.exception(self.node_name, exc_info=1)
    return wrapped


class OpencvRccorder(Codelet):
    def __init__(
            self, backend, logger, node_name, segmentation_out, color_out
        ):
        super().__init__(backend, logger, node_name)
        self.segmentation_out = segmentation_out
        self.color_out = color_out
        self.logger = logger
        self.node_name = node_name

    def start(self):
        self.log_info(
            "Start: %s segmentation_out: %s color_out: %s" %
            (self.node_name, self.segmentation_out, self.color_out)
        )
        self.rx_color = self.isaac_proto_rx("ColorCameraProto", "color")
        self.rx_segmentation = self.isaac_proto_rx("SegmentationCameraProto", "segmentation")
        self.color_writer = None
        self.segmentation_writer = None
        self.synchronize(self.rx_color, self.rx_segmentation)
        self.tick_on_message(self.rx_color)

    def stop(self):
        self.log_info("Releasing writers..")
        self.color_writer.release()
        self.segmentation_writer.release()

    def write_color(self, rx):
        proto = rx.get_proto()
        image = np.frombuffer(
            rx.get_buffer_content(
                proto.image.dataBufferIndex
            ), np.uint8).reshape(
                proto.image.rows, 
                proto.image.cols, 
                proto.image.channels
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.color_writer is None:
            self.color_writer = cv2.VideoWriter(
                self.color_out, cv2.VideoWriter_fourcc(*"mp4v"), 
                25.0, (proto.image.cols, proto.image.rows)
            )
        self.color_writer.write(image)

    def write_segmentation(self, rx):
        proto = rx.get_proto()
        image = np.frombuffer(
            rx.get_buffer_content(
                proto.labelImage.dataBufferIndex
            ), np.uint8).reshape(
                proto.labelImage.rows, 
                proto.labelImage.cols, 
                proto.labelImage.channels
            )
        
        # HACK: scale as we have only one floor label
        image = image*255
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.segmentation_writer is None:
            self.segmentation_writer = cv2.VideoWriter(
                self.segmentation_out, cv2.VideoWriter_fourcc(*"mp4v"), 
                25.0, (proto.labelImage.cols, proto.labelImage.rows)
            )
        self.segmentation_writer.write(image)


    @handleTickException
    def tick(self):
        assert self.rx_segmentation.available() and self.rx_color.available()
        self.write_color(self.rx_color)
        self.write_segmentation(self.rx_segmentation)

@click.command()
@click.option('--app_filename', required=True, type=click.Path(exists=True), help='Application(app_filename=..)')
@click.option('--more', default=(), type=click.Path(exists=True), help='Application(more=..)', multiple=True)
@click.option('--color_out', required=True, type=click.Path(resolve_path=True, writable=True),)
@click.option('--segmentation_out', required=True, type=click.Path(resolve_path=True, writable=True),)
def main(app_filename, more, color_out, segmentation_out):
    app = Application(app_filename=app_filename, more_jsons=','.join(more))
    app.register(
        {"opencv_recorder_node": partial(OpencvRccorder, color_out=color_out, segmentation_out=segmentation_out)}
    )
    app.start_wait_stop()


if __name__ == '__main__':
    main()
