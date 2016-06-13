// YOU CAN ALWAYS STOP THE DRONE BY TURNING IT AROUND

var client = require('ar-drone').createClient();	// get the drone client
const myaddon = require('../build/Release/MSC_drone');	// get the addon
var addon = myaddon();	// The addon function
var fs = require('fs');


var pngStream = client.getPngStream();

var lastPng;
pngStream
  .on('error', console.log)


var timerId;
const interval = 400;		// Timer interval between acquiring data from drone, in ms
setTimeout(function () { } , 1000);	// Wait for 1s

// Emergency landing, as long as the program has no error and does not quit
process.on('SIGINT', function () {
    console.log('Got SIGINT. Landing, press Control-C again to force exit.');
	//clearInterval(timerId);
	client.land();
    setTimeout(function () {
        console.log("Landing done.");
		console.log("battery level:",client.battery());
        //client.land();
        process.exit(0);
    }, 1000);
});


// Set up the transformation set variable
var finalTrans = {
	nc : -1,xt : 0,yt : 0,rot : 0,sc : 0
};

// Take off
client
    .takeoff();
	
// Use bottom camera
client.config('video:video_channel', 3);

// Control
client
	.after(1000,function(){
		client.up(1);
		console.log("up");
	});
client
    .after(2000,function() {
    if (1) {
        timerId = setInterval(function (lastPng) {
            console.log("once");
            pngStream		// Get data from pngStream for once
				.once('data', function (pngBuffer) {
						console.time("C time");
						lastPng = pngBuffer;
						var ctrlData = addon(lastPng,finalTrans.xt,finalTrans.yt,finalTrans.rot,finalTrans.sc,finalTrans.nc);
						//console.log(ctrlData);
						if (ctrlData.roll > 0) {
							client.left(ctrlData.roll);
						}
						else if (ctrlData.roll < 0) {
							client.right(-ctrlData.roll);
						}
						if (ctrlData.pitch > 0) {
							client.back(ctrlData.pitch);
						}
						else if (ctrlData.pitch < 0) {
							client.front(-ctrlData.pitch);
						}
						if (~ctrlData.roll&&~ctrlData.pitch) {
							client.stop();
						}
						finalTrans.xt=ctrlData.xt;
						finalTrans.yt = ctrlData.yt;
						finalTrans.rot = ctrlData.rot;
						finalTrans.sc = ctrlData.sc;
						finalTrans.nc = ctrlData.nc;
						console.timeEnd("C time");
					})
				}, interval);
    };
});






//fs.writeFile('./logo.png', lastPng, 'binary', function(err){
//            if (err) throw err
//            console.log('File saved.')
//        })

//var vdata;
// var buf = new Buffer ([0x62,0x75,0x66,0x66,0x65,0x72]);
//var buf = new Buffer ([1,2,3,4,5]);
//imTest(buf);

// video.on('data', imTest.imTest);
// console.log('This shoul');

//console.log(addon.isbuf(lastPng));

