// Initialize map with base layers
const cities = L.layerGroup();
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
});
const osmHOT = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>'
});
const openTopoMap = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
});
const CartoDbDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/{variant}/{z}/{x}/{y}{r}.png', {
    maxZoom: 20,
    variant: 'dark_all',
    attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
});
const CartoDbVoyager = L.tileLayer('https://{s}.basemaps.cartocdn.com/{variant}/{z}/{x}/{y}{r}.png', {
    maxZoom: 20,
    variant: 'rastertiles/voyager',
    attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
});
const CartoDbPositron = L.tileLayer('https://{s}.basemaps.cartocdn.com/{variant}/{z}/{x}/{y}{r}.png', {
    maxZoom: 20,
    variant: 'light_all',
    attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
});
const EsriWorldImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    variant: 'World_Imagery', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});

const EsriWorldStreetMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    variant: 'World_Street_Map', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriTopoMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    variant: 'World_Topo_Map', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriWorldTerrainMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 8,
    variant: 'World_Terrain_Base', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriWorldPhyiscalnMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 8,
    variant: 'World_Physical_Map', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriOceanBaseMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 10,
    variant: 'Ocean/World_Ocean_Base', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriNatGeoWorldMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 12,
    variant: 'NatGeo_World_Map', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const EsriWorldGrayCanvas = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 12,
    variant: 'Canvas/World_Light_Gray_Base', // Replace with the desired variant
    attribution: 'Map data: &copy; <a href="https://server.arcgisonline.com">ArcGIS</a>'
});
const map = L.map('map', {
    center: [24, 78],
    zoom: 4,
    layers: [osm, cities]
});

const baseLayers = {
    'OpenStreetMap': osm,
    'OpenStreetMap.HOT': osmHOT,
    'OpenTopoMap': openTopoMap,
    'CartoDb Dark': CartoDbDark,
    'CartoDb Voyager': CartoDbVoyager,
    'CartoDb Positron': CartoDbPositron,
    'ESRI World imagery': EsriWorldImagery,
    'ESRI World StreetMap': EsriWorldStreetMap,
    'ESRI TopoMap': EsriTopoMap,
    'ESRI TerrainMap': EsriWorldTerrainMap,
    'ESRI PhyiscalMap': EsriWorldPhyiscalnMap,
    'ESRI OcenBaseMap': EsriOceanBaseMap,
    'ESRI NatGeo WorldMap': EsriNatGeoWorldMap,
    'ESRI Gray Canvas': EsriWorldGrayCanvas
};

const overlays = {
    'overlay': cities
};

const layerControl = L.control.layers(baseLayers, overlays).addTo(map);

let currentPopup = null;
let userClickPopup = false;

// Handle map clicks
map.on('click', function (e) {
    if (currentPopup) {
        map.closePopup(currentPopup);
    }
    userClickPopup = true;

    currentPopup = L.popup()
        .setLatLng(e.latlng)
        .setContent('<p>Extract Watershed</p><button onclick="delineate(' + e.latlng.lat + ',' + e.latlng.lng + ')">Delineate</button>')
        .openOn(map);
});

// Update lat and lon on mousemove
map.on("mousemove", function (e) {
    const zoom = map.getZoom();
    document.getElementById("zoom").value = zoom;

    let numdigits = 3;
    if (zoom < 10) numdigits = 2;
    if (zoom < 7) numdigits = 1;
    if (zoom < 5) numdigits = 0;

    document.getElementById("mouselat").value = e.latlng.lat.toFixed(numdigits);
    document.getElementById("mouselon").value = e.latlng.lng.toFixed(numdigits);
});

let currentBasinLayer = null;
let currentStreamLayer = null;
let ouletLayer = null;

// Delineate function
function delineate(lat, lng) {
    map.closePopup(currentPopup);
    currentPopup = null;
    userClickPopup = false;

    document.getElementById('loading-bar').classList.add('active');

    if (currentBasinLayer) map.removeLayer(currentBasinLayer);
    if (currentStreamLayer) map.removeLayer(currentStreamLayer);
    if (ouletLayer) map.removeLayer(ouletLayer);

    $.ajax({
        url: 'http://localhost:5000/', // Replace with your backend URL
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ x: lng, y: lat }),
        success: function (response) {
            const basin = response[0];
            const streams = response[1];
            const outlet = response[2];


            // Create a GeoJSON layer with the custom star icon
            ouletLayer = L.geoJSON(outlet)

            currentBasinLayer = L.geoJSON(basin, {
                style: function (feature) {
                    return {
                        color: 'blue',
                        weight: 2,
                        opacity: 0.6
                    };
                },
                onEachFeature: function (feature) {
                    document.getElementById('a').value = feature.properties.area;
                    document.getElementById('p').value = feature.properties.perimeter;
                    document.getElementById('lb').value = feature.properties.basin_length;
                    document.getElementById('Dd').value = feature.properties.Dd;
                    document.getElementById('F').value = feature.properties.F;
                    document.getElementById('Rc').value = feature.properties.Rc;
                    document.getElementById('Re').value = feature.properties.Re;
                }
            }).addTo(map);

            currentStreamLayer = L.geoJSON(streams, {
                style: function (feature) {
                    return {
                        color: 'blue',
                        weight: 1,
                        opacity: 0.8
                    };
                },
                onEachFeature: function (feature, layer) {
                    layer.feature = feature;
                }
            }).addTo(map);

            map.fitBounds(currentBasinLayer.getBounds());
            document.getElementById('loading-bar').classList.remove('active');
        },
        error: function (xhr, status, error) {
            console.error('Error:', xhr, status, error);
            alert('Error occurred while delineating watershed.');
            document.getElementById('loading-bar').classList.remove('active');
        }
    });
}

// Check proximity to streams
function checkProximityToStreams(latlng) {
    let nearestStream = null;
    let nearestDistance = Infinity;

    currentStreamLayer.eachLayer(function (layer) {
        const streamLatLngs = layer.getLatLngs();
        streamLatLngs.forEach(function (streamLatLng) {
            const distance = latlng.distanceTo(streamLatLng);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestStream = layer;
            }
        });
    });

    return { nearestStream, nearestDistance };
}

// Handle mousemove for proximity check
map.on('mousemove', function (e) {
    if (currentStreamLayer) {
        const proximity = checkProximityToStreams(e.latlng);
        if (proximity.nearestStream && proximity.nearestDistance < 50) {
            const feature = proximity.nearestStream.feature;
            if (!userClickPopup) {
                L.popup()
                    .setLatLng(e.latlng)
                    .setContent('Stream Length: ' + feature.properties.length + ' Kms' + '<br>' +
                        'Stream Order: ' + feature.properties.stream_order
                    )
                    .openOn(map);
            }
        } else {
            if (!userClickPopup) {
                map.closePopup();
            }
        }
    }
});

// Handle download button click
document.getElementById('button').addEventListener('click', function () {
    if (currentBasinLayer && currentStreamLayer) {
        const basinData = currentBasinLayer.toGeoJSON();
        const streamData = currentStreamLayer.toGeoJSON();

        $.ajax({
            url: 'http://localhost:5000/download',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ basin: basinData, streams: streamData }),
            success: function (response) {
                const link = document.createElement('a');
                link.href = response.download_url;
                link.download = 'watershed_shapefiles.zip';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            },
            error: function (xhr, status, error) {
                console.error('Error:', xhr, status, error);
                alert('Error occurred while generating shapefile.');
            }
        });
    } else {
        alert('No basin or stream data available for download.');
    }
});
// Handle download button click
document.getElementById('button1').addEventListener('click', function () {
    if (currentBasinLayer && currentStreamLayer && ouletLayer) {
        const basinData = currentBasinLayer.toGeoJSON();
        const streamData = currentStreamLayer.toGeoJSON();
        const outletdata = ouletLayer.toGeoJSON();

        $.ajax({
            url: 'http://localhost:5000/download_csv',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ basin: basinData, streams: streamData, outlet:outletdata }),
            xhrFields: {
                responseType: 'blob' // Expect a binary file response
            },
            success: function (response) {
                const url = window.URL.createObjectURL(new Blob([response]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', 'morphometry_report.csv');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            },
            error: function (xhr, status, error) {
                console.error('Error:', xhr, status, error);
                alert('Error occurred while generating CSV.');
            }
        });
    } else {
        alert('No basin or stream data available for download.');
    }
});

// Handle download button click
document.getElementById('button3').addEventListener('click', function () {
    if (currentBasinLayer && currentStreamLayer) {
        const basinData = currentBasinLayer.toGeoJSON();
        const streamData = currentStreamLayer.toGeoJSON();
        const outletdata = ouletLayer.toGeoJSON();

        $.ajax({
            url: 'http://localhost:5000/download_pdf',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ basin: basinData, streams: streamData, outlet: outletdata }),
            xhrFields: {
                responseType: 'blob' // Expect a binary file response
            },
            success: function (response) {
                const url = window.URL.createObjectURL(new Blob([response]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', 'morphometry_report.pdf');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            },
            error: function (xhr, status, error) {
                console.error('Error:', xhr, status, error);
                alert('Error occurred while generating CSV.');
            }
        });
    } else {
        alert('No basin or stream data available for download.');
    }
});