
Jmol._isAsync = false;
var jmolApplet0;

var Info = {
    width: 450,
    height: 450,
    debug: false,
    color: "0xFFFFFF",
    addSelectionOptions: false,
    use: "HTML5",
    // JAVA HTML5 WEBGL are all options
    j2sPath: "/static/jsmol/j2s",
    // XXX how coded for now.
    //serverURL: "http://chemapps.stolaf.edu/jmol/jsmol/php/jsmol.php",
    readyFunction: jmol_isReady,
    disableJ2SLoadMonitor: true,
    disableInitialConsole: true,
    allowJavaScript: false
};


function repeatCell(n1, n2, n3)
{
    var s = '{ ' + n1.toString() + ' ' + n2.toString() + ' ' + n3.toString() +
        ' };';
    Jmol.script(jmolApplet0, 'load "" ' + s);
}

$(document).ready(function()
{
    $("#appdiv").html(Jmol.getAppletHtml("jmolApplet0", Info))
})
