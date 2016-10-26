// YOU CAN ALWAYS STOP THE DRONE BY TURNING IT AROUND

var client = require('ar-drone').createClient();	// get the drone client
const myaddon = require('../build/Release/MSC_drone');	// get the addon
var addon = myaddon();	// The addon function
var fs = require('fs');
var async = require('async');


var pngStream = client.getPngStream();

var lastPng;
pngStream.on('error', console.log);

const interval = 400;
setTimeout(function () { } , 1000);

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
/* finalTrans.nc = -1;
finalTrans.xt = 0;
finalTrans.yt = 0;
finalTrans.rot = 0;
finalTrans.sc = 0; */

var mainLoop = function () {
    setInterval(function (lastPng) {
        console.log("once");
        pngStream
            .once('data', function (pngBuffer) {
                console.time("C time");
                lastPng = pngBuffer;
                //console.log("It is a buffer:" + Buffer.isBuffer(lastPng));
                //console.log(addon.myctrl(lastPng));
                var ctrlData = addon(lastPng,finalTrans.xt,finalTrans.yt,finalTrans.rot,finalTrans.sc,finalTrans.nc);
                console.log(ctrlData);
                
                if (ctrlData.roll > 0) {
                    client.left(ctrlData.roll);
                }
                else if (ctrlData.roll < 0) {
                    client.right(-ctrlData.roll);
                }

                if (ctrlData.pitch > 0) {
                    client.front(ctrlData.pitch);
                }
                else if (ctrlData.pitch < 0) {
                    client.back(-ctrlData.pitch);
                }
                
                if (ctrlData.lift > 0) {
                    client.up(ctrlData.lift);
                }
                else if (ctrlData.lift < 0) {
                    client.down(-ctrlData.lift);
                }
                
                if (~ctrlData.roll&&~ctrlData.pitch&&~ctrlData.lift) {
                    client.stop();
                }
                finalTrans.xt=ctrlData.xt;
                finalTrans.yt = ctrlData.yt;
                finalTrans.rot = ctrlData.rot;
                finalTrans.sc = ctrlData.sc;
                finalTrans.nc = ctrlData.nc;
                console.timeEnd("C time");
            });
        }, 
    interval);
};

async.waterfall([
    function (callback) {
        var options = {
            key : 'video:video_channel',
            value : 0,
            timeout : 1000
        };
        console.log("Step 1");
        client.config(options, callback);
    },
    function (callback) {
        console.log("Step 2");
        client.takeoff();
        setTimeout(function () { callback(null); }, 1500);
    },
    function (callback) {
        console.log("Step 3");
        mainLoop();
    }
], function (err, result) {
    if (err) {
        console.log(err);
    }
});