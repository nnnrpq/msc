// Run this to receive the raw video stream from your drone as buffers.

var arDrone = require('ar-drone');

var video = arDrone.createClient().getVideoStream();

video.on('data', console.log);
video.on('error', console.log);
