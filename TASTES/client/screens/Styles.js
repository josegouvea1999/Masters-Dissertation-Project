import { StyleSheet } from "react-native";

const styles = StyleSheet.create({
    userBar: {
      marginTop: 30,
      padding: 10,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "left"
    },
    userIcon: {
      width: 60,
      height: 60,
      borderRadius: 30,
      resizeMode: "cover",
      borderWidth: 3,
      borderColor: "#1DB954",
      marginLeft: 10
    },
    userMessage: {
      marginLeft: 15,
      fontSize: 20,
      fontWeight: "bold",
      color: "#1DB954",
    },
    selectPlaylistMessage: {
      marginLeft: 15,
      fontSize: 12,
      color: "white",
    },
    centeredView: {
      flex: 1,
      justifyContent: 'flex-end',
    },
    modalView: {
      backgroundColor: '#1DB954',
      borderTopLeftRadius: 30,
      borderTopRightRadius: 30,
      padding: 20,
      paddingBottom: 30,
    },
    modalContainer: {
      width: '100%',
      height: '100%',
      backgroundColor: 'transparent',
      justifyContent: 'flex-end',
    },
    textStyle: {
      fontSize: 20,
      color: 'white',
      fontWeight: 'bold',
      textAlign: 'center',
    },
    textStyleExpanding: {
      fontSize: 18,
      color: 'white',
      fontWeight: 'bold',
      textAlign: 'center',
    },
    textStyleSuccess: {
      fontSize: 24,
      color: 'white',
      fontWeight: 'bold',
      textAlign: 'center',
      marginTop: 20,
    },
    textStyleSuccess2: {
      fontSize: 15,
      color: 'white',
      fontWeight: 'bold',
      textAlign: 'center',
      maxWidth: '80%',
      marginTop: 10,
    },
    textStyleSuccess3: {
      fontSize: 16,
      color: 'white',
      fontWeight: 'bold',
      textAlign: 'center',
      marginTop: 10,
      maxWidth: '70%'
    },
    modalText: {
      marginBottom: 15,
    },
    modalTextExpanding: {
      marginTop: 50, 
      marginBottom: 15,
    },
    playlistInfoContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: 10,
      marginTop: 10,
    },
    playlistImageModal: {
      width: 80,
      height: 80,
      borderRadius: 5,
    },
    playlistImageModalExpanding: {
      width: 200,
      height: 200,
      borderRadius: 10,
    },
    playlistImageModalSuccess: {
      width: 120,
      height: 120,
      borderRadius: 10,
      marginTop: 40,
    },
    playlistNameModal: {
      color: 'white',
      fontSize: 18,
      fontWeight: 'bold',
      marginLeft: 15,
      maxWidth: '70%',
    },
    buttonContainer: {
      flexDirection: 'row',
      justifyContent: 'center',
      marginTop: 30
    },
    buttonContainerSuccess: {
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      marginTop: 30,
      width: '90%',
    },
    button: {
      borderRadius: 20,
      paddingVertical: 8,
      paddingHorizontal: 30,
      flex: 1,
      
    },
    buttonSuccess: {
      borderRadius: 20,
      paddingVertical: 8,
      paddingHorizontal: 30,
      alignSelf: 'stretch',
      marginVertical: 10,
    },
    buttonClose: {
      backgroundColor: 'transparent',
    },
    cancelButton: {
      borderColor: 'white',
      borderWidth: 1,
    },
    confirmButton: {
      backgroundColor: 'white',
      marginLeft: 10,
    },
    confirmButtonSuccess: {
      backgroundColor: 'white',
    },
    confirmButtonText: {
      color: 'black',
      fontSize: 16,
    },
    cancelButtonText: {
      color: 'white',
      fontSize: 16,
    },
    loadingContainer: {
      flex: 1,
      justifyContent: "center",
      alignItems: "center",
    },
    loadingLogo: {
      width: 150,
      height: 150,
      resizeMode: "contain",
    },
    loadingLogoExpanding: {
      width: 200,
      height: 200,
      resizeMode: "contain",
    },
    loadingText: {
      color: "white",
      fontSize: 16,
      marginTop: 10,
    },
    playlistPressable: {
      marginBottom: 10,
      flexDirection: "row",
      alignItems: "center",
      gap: 10,
      flex: 1,
      marginHorizontal: 10,
      marginVertical: 3,
      backgroundColor: "#202020",
      borderRadius: 4,
      elevation: 3,  
    },
    coverPressable: {
      width: 75,
      height: 75,
      justifyContent: "center",
      alignItems: "center",
    },
    playlistName: {
      color: "white", 
      fontSize: 15, 
      fontWeight: "bold", 
      maxWidth: '80%',
    },
    expandingContent: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 20,
    },
    normalContent: {
      flex: 1,
      alignItems: 'center',
    },
    overlayContainer: {
      ...StyleSheet.absoluteFillObject,
      alignItems: 'center',
      justifyContent: 'center',
    },
    pulsatingImage: {
      width: 200,
      height: 200,
      position: 'absolute',
      zIndex: 1,
    },
    container: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      marginVertical: 10,
    },
    addButton: {
      width: 50,
      height: 50,
      borderRadius: 40,
      backgroundColor: '#1DB954',
      justifyContent: 'center',
      alignItems: 'center',
    },
    addButtonText: {
      fontSize: 25,
      color: 'white',
      bottom: 1,
    },
  });

  export {styles};