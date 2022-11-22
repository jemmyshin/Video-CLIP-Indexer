import av


content = '/Users/jemfu/Desktop/1668665259003035.mp4'
with av.open(content) as container:
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'
    for frame in container.decode(stream):
        frame.to_image().save(
            'frame.{:04d}.png'.format(frame.pts),
            quality=80,
        )