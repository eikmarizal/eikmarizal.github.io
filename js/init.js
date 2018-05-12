$('.button-collapse').sideNav();
$('.collapsible').collapsible();
$('select').material_select();

function setgraph(name){
  var list_rgdpe = (country_name_rg.indexOf(name) > 0 ? rgdpe[country_name_rg.indexOf(name)] : new Array(4).fill(0));
  var list_rgdpo = (country_name_rg.indexOf(name) > 0 ? rdgpo[country_name_rg.indexOf(name)] : new Array(4).fill(0));
  var list_pop = (country_name_rg.indexOf(name) > 0 ? pop[country_name_rg.indexOf(name)] : new Array(4).fill(0));
  var list_ccon = (country_name_rg.indexOf(name) > 0 ? ccon[country_name_rg.indexOf(name)] : new Array(4).fill(0));
  var list_avg = (country_name_rg.indexOf(name) > 0 ? avg[country_name_rg.indexOf(name)] : new Array(4).fill(0));
  var umployment = {
    x: ["2011", "2012", "2013", "2014"],
    y: stats[country_name.indexOf(name)],
    mode: 'lines',
    name: 'Unemployment rate',
    type: 'scatter'
  };
  var rgdpe_ = {
    x: ["2011", "2012", "2013", "2014"],
    y: list_rgdpe,
    mode: 'lines',
    name: 'Expenditure-side (in mil) E+4',
    type: 'scatter'
  };
  var rgdpo_ = {
    x: ["2011", "2012", "2013", "2014"],
    y: list_rgdpo,
    mode: 'lines',
    name: 'Output-side (in mil) E+3',
    type: 'scatter'
  };
  var pop_ = {
    x: ["2011", "2012", "2013", "2014"],
    y: list_pop,
    mode: 'lines',
    name: 'Population (in millions)',
    type: 'scatter'
  };
  var ccon_ = {
    x: ["2011", "2012", "2013", "2014"],
    y: list_ccon,
    mode: 'lines',
    name: 'Real consumption of households and government (in mil) E+3',
    type: 'scatter'
  };
  var avg_ = {
    x: ["2011", "2012", "2013", "2014"],
    y: list_avg,
    mode: 'lines',
    name: 'Average annual hours',
    type: 'scatter'
  };
  var layout = {
    title: 'Unemployment rate for ' + name,
    xaxis:{
      title: 'Year',
      autotick: false
    },
    yaxis:{
      title: 'Values'
    }
  };
  Plotly.newPlot('myDiv3', [umployment, rgdpe_, rgdpo_, pop_, ccon_, avg_], layout);
}

var data_2010 = [{
  type: 'choropleth',
  autocolorscale: false,
  colorscale: metric,
  showscale: true,
  locations: country_name,
  z: stat_2010,
  locationmode: 'country names',
  text: country_name,
  marker:{
    line:{
      color:'rgb(250,250,200)',
      width: 0.5
    },
    colorbar:{
      autotick: true,
      tickprefix: ''
    },
    title: 'Unemployment Rate'
  }
}];

var layout_2010 = {
  title: 'World Map of Global Youth Unemployment in the Year 2010',
  geo:{
    showframe: true,
    showocean: true,
    oceancolor: 'rgb(28,107,160)',
    projection:{
      type: 'orthographic',
      rotation:{
        lon: 60,
        lat: 10
      }
    },
    lonaxis:{
      showgrid: false,
      gridcolor: 'rgb(202, 202, 202)',
      width: '0.05'
    },
    lataxis:{
      showgrid: false,
      gridcolor: 'rgb(102, 102, 102)'
    }
  }
}

Plotly.newPlot('myDiv', data_2010, layout_2010);
var myPlot = document.getElementById('myDiv2');
var data_2014 = [{
  type: 'choropleth',
  autocolorscale: false,
  colorscale: metric,
  showscale: true,
  locations: country_name,
  z: stat_2014,
  locationmode: 'country names',
  text: country_name,
  marker:{
    line:{
      color:'rgb(250,250,200)',
      width: 0.5
    },
    colorbar:{
      autotick: true,
      tickprefix: ''
    },
    title: 'Unemployment Rate'
  }
}];

var layout_2014 = {
  title: 'World Map of Global Youth Unemployment in the Year 2014, click me!',
  geo:{
    showframe: true,
    showocean: true,
    oceancolor: 'rgb(28,107,160)',
    projection:{
      type: 'orthographic',
      rotation:{
        lon: 60,
        lat: 10
      }
    },
    lonaxis:{
      showgrid: false,
      gridcolor: 'rgb(202, 202, 202)',
      width: '0.05'
    },
    lataxis:{
      showgrid: false,
      gridcolor: 'rgb(102, 102, 102)'
    }
  }
}

Plotly.newPlot('myDiv2', data_2014, layout_2014);
myPlot.on('plotly_click', function(notate){
  setgraph(notate.points[0]['text']);
});
setgraph("Afghanistan");
